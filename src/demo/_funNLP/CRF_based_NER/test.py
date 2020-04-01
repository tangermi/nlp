from ner import CorpusProcess

if __name__ == '__main__':
    process = CorpusProcess()
    text = '19980101-01-001-006/m  在/p  １９９８年/t  来临/v  之际/f  ，/w  我/r  十分/m  高兴/a  地/u  通过/p  [中央/n  人民/n  广播/vn  电台/n]nt  、/w  [中国/ns  国际/n  广播/vn  电台/n]nt  和/c  [中央/n  电视台/n]nt  ，/w  向/p  全国/n  各族/r  人民/n  ，/w  向/p  [香港/ns  特别/a  行政区/n]ns  同胞/n  、/w  澳门/ns  和/c  台湾/ns  同胞/n  、/w  海外/s  侨胞/n  ，/w  向/p  世界/n  各国/r  的/u  朋友/n  们/k  ，/w  致以/v  诚挚/a  的/u  问候/vn  和/c  良好/a  的/u  祝愿/vn  ！/w'
    print(text)
    print(text.strip())
    first_step = process.q_to_b(text).strip().split(u'  ')
    print(first_step)
    second_step = process.process_t(first_step)
    print(second_step)
    third_step = process.process_nr(second_step)
    print(third_step)
    final_step = process.process_k(third_step)
    print(final_step)
    new_line = '  '.join(final_step[1:])
    print(new_line)
