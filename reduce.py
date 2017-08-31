import codecs
import argparse
import sys

def reduce_dataset(fin, fout, max_allowed_word_len=30):
    """
    If a line contains a word longer than max allowed length, ignore that line
    :param fin: file to read
    :param fout: file to write
    :param max_allowed_word_len: maximum word length to write the line
    """
    input = codecs.open(fin, mode='r', encoding='utf-8')
    output = codecs.open(fout, mode='w', encoding='utf-8')
    lncnt = 0
    with input,output:
        for line in input:
            line = line.strip()
            wd = True
            for word in line.split():
                if len(word) > max_allowed_word_len:
                    wd = False
                    break
            if wd:
                lncnt+=1
                output.write(line+'\n')
    print "Number of lines ",lncnt

# parse morphologically disambiguated
def parse_morph_dis(fin, fout, START_TAG=u'<S>',END_TAG=u'</S>',SEP=u' '):
    """
    Parse MD dataset - save as sentences
    :param fin: file to read
    :param fout: file to write
    :param START_TAG='<S>': sentence start
    :param END_TAG='<S>': sentence end
    :param SEP=' ': seperator
    """
    input = codecs.open(fin, mode='r', encoding='utf-8-sig')
    output_sent = codecs.open(fout, mode='w', encoding='utf-8')
    output_sent_morph = codecs.open(fout+".morph", mode='w', encoding='utf-8')
    sentcnt = 0
    with input,output_sent,output_sent_morph:
        for line in input:
            line = line.strip()
            parts = line.split(SEP)
            wrd = parts[0]
            correct_tag = parts[1]
            if wrd==START_TAG:
                sent = []
                sent_morph = []
                continue
            elif wrd==END_TAG:
                lsent = " ".join(sent)
                lmorph = " ".join(sent_morph)
                output_sent.write(lsent+'\n')
                output_sent_morph.write(lmorph+'\n')
                sentcnt+=1
                continue
            # if it is another tag
            elif wrd.startswith(u'<'):
                continue
            else:
                sent.append(wrd)
                sent_morph.append(correct_tag)
    print "Number of sentences ",sentcnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', type=str, default='/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/allData_new.cln',
                        help="Raw clean text")
    parser.add_argument('--fout', type=str, default='/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/reduced.txt',
                        help="Out file with reduced set of sentences")
    parser.add_argument('--maxlen', type=int, default=30,
                        help="Max length of a word in a sentence")
    parser.add_argument('--op', type=int, default=1,
                        help="1=reduce, 2=parse morph disamb")

    args = parser.parse_args()
    if args.op==1:
        reduce_dataset(args.fin, args.fout, args.maxlen)
    elif args.op==2:
        parse_morph_dis(args.fin, args.fout)
    else:
        sys.exit("Wrong option number")

if __name__ == '__main__':
    main()