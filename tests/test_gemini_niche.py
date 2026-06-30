# tests/test_gemini_niche.py
import tools.gemini_narrative_pass as g


def test_system_prompt_includes_niche_register_when_set():
    # the assembler should append the register block; expose it as a small helper
    sys_with = g._append_niche("BASE SYSTEM", niche="C", niche_secondary="A")
    assert "Dark-Action/Revenge" in sys_with and "SECONDARY" in sys_with
    assert g._append_niche("BASE SYSTEM", "", "") == "BASE SYSTEM"  # no-op when unset
