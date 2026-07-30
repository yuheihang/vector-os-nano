# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for unified NavigateSkill -- hardware-agnostic navigation."""
import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.types import SkillResult


def _make_scene_graph():
    """Create a SceneGraph loaded with room_layout.yaml for navigate tests."""
    import os
    from vector_os_nano.core.scene_graph import SceneGraph
    sg = SceneGraph()
    layout = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "room_layout.yaml",
    )
    if os.path.isfile(layout):
        sg.load_layout(layout)
    # Mark all rooms as visited (visit_count >= 1) so navigate trusts positions
    for room_id in list(sg._rooms):
        r = sg._rooms[room_id]
        sg._rooms[room_id] = type(r)(
            room_id=r.room_id, center_x=r.center_x, center_y=r.center_y,
            area=r.area, visit_count=1, last_visited=1.0,
            representative_description=r.representative_description,
            connected_rooms=r.connected_rooms,
        )
    return sg


def _make_context(with_nav=False):
    base = MagicMock()
    base.walk.return_value = True
    base.get_position.return_value = [10.0, 3.0, 0.27]
    base.get_heading.return_value = 0.0
    base.navigate_to = MagicMock()

    def _navigate_to(x, y, **_kwargs):
        # A successful FAR contract means the requested segment actually landed.
        # Keeping position stale would correctly trip P2's fail-closed tolerance
        # check and is not a valid navigation fake.
        base.get_position.return_value = [float(x), float(y), 0.27]
        return True

    base.navigate_to.side_effect = _navigate_to
    sg = _make_scene_graph()
    services = {"spatial_memory": sg}
    if with_nav:
        nav = MagicMock()
        nav.is_available = True
        nav.navigate_to.return_value = True
        services["nav"] = nav
    return SkillContext(
        bases={"go2": base},
        world_model=WorldModel(),
        services=services,
    )


class TestNavigateSkillMetadata:
    def test_skill_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert skill.name == "navigate"

    def test_aliases(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert "去" in skill.__class__.__skill_aliases__
        assert "navigate" in skill.__class__.__skill_aliases__

    def test_has_required_metadata(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert hasattr(skill, "description")
        assert hasattr(skill, "parameters")
        assert "room" in skill.parameters
        assert hasattr(skill, "failure_modes")
        assert "no_base" in skill.failure_modes
        assert "unknown_room" in skill.failure_modes
        assert "navigation_failed" in skill.failure_modes


class TestNavigateWithNavStack:
    def test_proxy_navigate_to_used_when_available(self):
        """Executable v2 layout expands the room goal into FAR segments."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        # hallway -> kitchen is pre/center/post + room_goal; the direct nav
        # service is never used in known-layout v2 mode.
        assert [call.args[:2] for call in ctx.base.navigate_to.call_args_list] == [
            (13.4, 3.0),
            (14.0, 3.0),
            (14.6, 3.0),
            (17.0, 2.5),
        ]
        assert all(
            call.kwargs["allow_door_fallback"] is False
            for call in ctx.base.navigate_to.call_args_list
        )
        ctx.services["nav"].navigate_to.assert_not_called()

    def test_proxy_receives_kitchen_coordinates(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        args = ctx.base.navigate_to.call_args
        # Kitchen center is (17.0, 2.5)
        assert 15 < args[0][0] < 19
        assert 1 < args[0][1] < 4

    def test_proxy_segment_failure_does_not_fall_back_or_skip(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        ctx.base.navigate_to.side_effect = lambda *_args, **_kwargs: False
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert result.diagnosis_code == "segment_no_path"
        assert ctx.base.navigate_to.call_count == 1
        ctx.services["nav"].navigate_to.assert_not_called()

    def test_nav_unavailable_falls_back_to_dead_reckoning(self):
        """nav service present but is_available=False -> falls back to dead-reckoning."""
        from vector_os_nano.skills.navigate import NavigateSkill
        nav = MagicMock()
        nav.is_available = False
        nav.navigate_to.return_value = True
        base = MagicMock(
            walk=MagicMock(return_value=True),
            get_position=MagicMock(return_value=[10.0, 3.0, 0.27]),
            get_heading=MagicMock(return_value=0.0),
            navigate_to=MagicMock(),
        )

        def _navigate_to(x, y, **_kwargs):
            base.get_position.return_value = [float(x), float(y), 0.27]
            return True

        base.navigate_to.side_effect = _navigate_to
        sg = _make_scene_graph()
        ctx = SkillContext(
            bases={"go2": base},
            world_model=WorldModel(),
            services={"nav": nav, "spatial_memory": sg},
        )
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        # Dead-reckoning path; nav.navigate_to must NOT be called
        nav.navigate_to.assert_not_called()

    def test_nav_unknown_room_not_sent_to_nav_client(self):
        """Unknown room should fail before reaching nav client."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "mars_base"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code
        ctx.services["nav"].navigate_to.assert_not_called()


class TestNavigateDeadReckoning:
    def test_fallback_without_nav_stack(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success

    def test_unknown_room(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "nonexistent_room"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_chinese_room_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "厨房"}, ctx)
        assert result.success

    def test_no_base(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = SkillContext(world_model=WorldModel())
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "no_base" in result.diagnosis_code

    def test_already_at_destination(self):
        """Robot already near target room center -- should succeed immediately."""
        from vector_os_nano.skills.navigate import NavigateSkill
        # Position (3.0, 2.5) is the living_room center
        base = MagicMock()
        base.walk.return_value = True
        base.get_position.return_value = [3.0, 2.5, 0.27]
        base.get_heading.return_value = 0.0
        base.navigate_to = MagicMock(return_value=True)
        sg = _make_scene_graph()
        ctx = SkillContext(bases={"go2": base}, world_model=WorldModel(),
                          services={"spatial_memory": sg})
        skill = NavigateSkill()
        result = skill.execute({"room": "living_room"}, ctx)
        assert result.success

    def test_dead_reckoning_result_data(self):
        """Dead-reckoning success should include room and position in result_data."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        assert result.result_data is not None
        assert "room" in result.result_data
        assert result.result_data["room"] == "kitchen"

    def test_alias_english_shorthand(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        # "bedroom" is an alias for master_bedroom
        result = skill.execute({"room": "bedroom"}, ctx)
        assert result.success
        assert result.result_data["room"] == "master_bedroom"

    def test_empty_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": ""}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_missing_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code


class TestNavigateFromGo2Package:
    def test_go2_skills_includes_navigate(self):
        from vector_os_nano.skills.go2 import get_go2_skills
        skills = get_go2_skills()
        names = {s.name for s in skills}
        assert "navigate" in names

    def test_go2_navigate_is_unified(self):
        """The NavigateSkill from go2 package should be the unified one."""
        from vector_os_nano.skills.go2 import NavigateSkill as Go2Nav
        from vector_os_nano.skills.navigate import NavigateSkill as UnifiedNav
        assert Go2Nav is UnifiedNav

    def test_go2_navigate_module_removed(self):
        """go2/navigate.py shim has been removed -- import must fail."""
        import importlib
        import pytest
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("vector_os_nano.skills.go2.navigate")
