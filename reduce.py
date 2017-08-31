import codecs
import argparse

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


# Reduce dataset
#fin = "/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/allData_new.cln"
#fout = "/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/reduced.txt"
#reduce_dataset(fin,fout)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', type=str, default='/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/allData_new.cln',
                        help="Raw clean text")
    parser.add_argument('--fout', type=str, default='/media/isguderg/Work/Doktora/Data Sets/LMSet_Clean/reduced.txt',
                        help="Out file with reduced set of sentences")
    parser.add_argument('--maxlen', type=int, default=30,
                        help="Max length of a word in a sentence")
    args = parser.parse_args()
    reduce_dataset(args.fin, args.fout, args.maxlen)

if __name__ == '__main__':
    main()