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
    assert stats == {"scored": 3, "with_quote": 2, "quotes": 2, "verified": 1, "downgraded": 2}


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
