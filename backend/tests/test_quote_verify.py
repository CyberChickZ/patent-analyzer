import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.quote_verify import locate_quote, verify_checklist_results

DOC = ("[0001] The apparatus comprises a virtual channel unit configured to time-share "
       "a serial bus between a first virtual channel and a second virtual channel.\n"
       "[0002] A buffering control unit receives data via the first virtual channel; "
       "the switching unit stores that data in the second receive buffer when the "
       "link partner lacks VC1 support.")


def test_exact_quote_found():
    found, sim = locate_quote("a buffering control unit receives data via the first virtual channel", DOC)
    assert found and sim == 1.0


def test_punctuation_and_case_insensitive():
    found, _ = locate_quote("Time-Share a Serial Bus, between a first virtual channel", DOC)
    assert found


def test_minor_ocr_noise_still_found():
    found, sim = locate_quote("the switching unit stores that data in the secnod receive buffer when", DOC)
    assert found and sim >= 0.9


def test_fabricated_quote_rejected():
    found, sim = locate_quote("the apparatus uses quantum tunneling to encrypt every packet", DOC)
    assert not found and sim < 0.9


def test_short_quote_rejected():
    assert locate_quote("serial", DOC) == (False, 0.0)


def test_verify_downgrades_unverifiable():
    cr = {
        "vc unit": {"score": 2, "evidence_quote": "virtual channel unit configured to time-share a serial bus"},
        "encrypt": {"score": 2, "evidence_quote": "packets are encrypted with a rotating key schedule"},
        "no quote": {"score": 1, "evidence_quote": ""},
        "absent": {"score": 0},
    }
    stats = verify_checklist_results(cr, DOC)
    assert cr["vc unit"]["score"] == 2
    assert cr["encrypt"]["score"] == 0 and cr["encrypt"]["quote_unverified"]
    assert cr["no quote"]["score"] == 0 and cr["no quote"]["match"] is False
    assert cr["absent"]["score"] == 0 and "quote_unverified" not in cr["absent"]
    assert stats == {"scored": 3, "with_quote": 2, "quotes": 2, "verified": 1, "downgraded": 2, "translated": 0}


def test_verify_keeps_multiple_quotes_and_downgrades_only_when_none_survive():
    cr = {"buffer": {"score": 2, "evidence_quotes": [
        "a buffering control unit receives data via the first virtual channel",
        "the switching unit stores that data in the second receive buffer",
        "packets are encrypted with a rotating key schedule"]}}
    stats = verify_checklist_results(cr, DOC)
    item = cr["buffer"]
    assert item["score"] == 2 and len(item["verified_quotes"]) == 2
    assert [c["verified"] for c in item["quote_checks"]] == [True, True, False]
    assert item["evidence_quote"] == item["verified_quotes"][0]
    assert stats["quotes"] == 3 and stats["verified"] == 2 and stats["downgraded"] == 0


# --- PDF-path failures seen in the demo job (H3) ---------------------------

from patent_analyzer.quote_verify import normalize

CN_DOC = ("(57)摘要\n本发明涉及一种基于上限置信区间算法的\n无人机群协同巡逻追踪轨迹规划方法：输入巡逻\n"
          "区域、无人机数量与加油站位置后，本发明将构\n建目标概率模型，然后用上限置信区间算法求出\n"
          "无人机下一步运动方向。\n步骤602、计数器t=1开始迭代，利用第(t‑1)轮迭代中的用户分组\n"
          "和无人机的位置\n，求解优化问题更新用户功率分配；\n")


def test_cjk_quote_across_pdf_line_breaks_and_fullwidth_punctuation():
    found, sim = locate_quote("本发明涉及一种基于上限置信区间算法的无人机群协同巡逻追踪轨迹规划方法:输入巡逻区域、无人机数量与加油站位置后", CN_DOC)
    assert found and sim == 1.0


def test_cjk_quote_from_other_document_rejected():
    found, sim = locate_quote("每架无人机都有一个对应的队友模型,每个模型记录该无人机的最大概率方向", CN_DOC)
    assert not found


def test_ligature_and_fullwidth_forms_normalized():
    assert normalize("ﬁnal conﬁguration，ＡＢＣ") == "final configuration abc"


from patent_analyzer.quote_dual import tokens, verify_quote_dual


def test_cjk_tokens_are_character_bigrams():
    assert tokens("无人机 t=1") == ["无人", "人机", "t", "1"]
    assert tokens("Für eine") == ["für", "eine"]


def test_cjk_quote_with_formula_dropped_by_pdf_passes_dual_not_locate():
    q = "步骤602、计数器t=1开始迭代,利用第(t-1)轮迭代中的用户分组{K_m(t-1)}和无人机的位置{v_m(t-1)},求解优化问题更新用户功率分配;"
    ok, sr, br = verify_quote_dual(q, CN_DOC)
    assert ok and sr >= 0.85 and br >= 0.7
    ok, _, _ = verify_quote_dual("每架无人机都有一个对应的队友模型,每个模型记录该无人机的最大概率方向", CN_DOC)
    assert not ok


from patent_analyzer.quote_verify import quote_segments, strip_labels


def test_source_labels_and_ellipsis_are_stripped():
    assert strip_labels("将摄像头拍摄的实时画面进行特征信息的提取; (Abstract)") == "将摄像头拍摄的实时画面进行特征信息的提取;"
    assert strip_labels("Claim 1: Initialize the single UAV") == "Initialize the single UAV"
    assert strip_labels("[Page 4] ...the input (o^n, d^n) of Q_g") == "...the input (o^n, d^n) of Q_g"
    assert strip_labels("对于待执行的任务; (Claim 1: For tasks (to be) executed)") == "对于待执行的任务;"
    assert quote_segments("a buffering control unit ... second receive buffer (Claim 2)") == \
        ["a buffering control unit", "second receive buffer"]


def test_elided_quote_needs_every_segment():
    found, _ = locate_quote("a buffering control unit receives data ... stores that data in the second receive buffer", DOC)
    assert found
    found, _ = locate_quote("a buffering control unit receives data ... encrypts every packet with a key", DOC)
    assert not found


def test_ligature_dash_and_hyphenless_line_break():
    pdf = "Enabling this\ncapability requires the ﬁnal conﬁguration of motion–LED behaviors while accompa\nnying agents"
    found, sim = locate_quote("the final configuration of motion-LED behaviors while accompanying agents", pdf)
    assert found and sim >= 0.9


from patent_analyzer.quote_verify import script_mismatch


def test_translated_quote_is_flagged_not_verified():
    doc = CN_DOC * 20
    q = "The invention relates to a UAV swarm cooperative patrol trajectory planning method based on the upper confidence bound"
    assert script_mismatch(q, doc)
    cr = {"x": {"score": 2, "evidence_quotes": [q]}}
    stats = verify_checklist_results(cr, doc)
    assert cr["x"]["score"] == 0 and cr["x"]["quote_checks"][0]["reason"] == "translated"
    assert stats["translated"] == 1 and stats["downgraded"] == 1
