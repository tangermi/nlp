from build import FullTextBuild
from search import FullTextSearch
"""
采用whoosh搜索，会将结果高亮，并打印bm25的分数
"""
if __name__ == "__main__":
    fulltextbuild = FullTextBuild('/home/xiaoxinwei/data/index')

    fulltextbuild.add_doc(title="document-1", path="/c", content="现在，我代表国务院，向大会报告政府工作，请予审议，并请全国政协委员提出意见。",
                          tags="speech", icon="/icon/fruit")
    fulltextbuild.add_doc(title="document-2", path="/a", content="报告工作，经济结构不断优化。消费拉动经济增长作用进一步增强。",
                          tags="economy", icon="/icon/fruit")
    fulltextbuild.add_doc(title="document-3", path="/b", content="深化供给侧结构性改革，实体经济活力不断释放。加大“破、立、降”力度。推进钢铁、煤炭行业市场化去产能。",
                          tags="economy", icon="/icon/fruit")

    fulltextsearch = FullTextSearch()
    fulltextsearch.load(fulltextbuild.index)
    results = fulltextsearch.search("经济")
    for hit in results:
        print(hit.highlights("content"))
        print(hit.score)