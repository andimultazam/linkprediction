import fnmatch
from os.path import isdir, join
from shutil import copytree
import os
import re
import Utils


def include_patterns(patterns):
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        dir_names = (name for name in all_names if isdir(join(path, name)))
        return set(all_names) - set(keep) - set(dir_names)
    return _ignore_patterns

def copy_files_with_patterns(src, dst, patterns):
    Utils.removeExistingDir(dst)
    copytree(src, dst, ignore=include_patterns(patterns))
    print('Finished copying files from "{}" to "{}"'.format(src, dst))

def list_ego_node_ids(path):
    regex = re.compile('[^a-zA-Z0-9\']+')
    filenames = os.listdir(path)
    for i in range(len(filenames)):
        filenames[i] = filenames[i].split(".")[0]
    return list(dict.fromkeys(filenames))

if __name__ == '__main__':
    import random

    #Input parameters
    # src = sys.argv[1]
    # dst = sys.argv[2]
    #pct = int(sys.argv[3])
    src = r'/home/kienguye/NUS/BigData/FinalProject/twitter'
    dst = r'/home/kienguye/NUS/BigData/FinalProject/twitter_t0.12'
    pct = 0.12

    #List all ego-node ids
    egoNodeIds = list_ego_node_ids(src)

    #Randomly select pct% of ego nodes.
    numEgoNodes = len(egoNodeIds)
    random.shuffle(egoNodeIds)
    print('No. of ego-nodes: {}'.format(numEgoNodes))
    numSelectedEgoNodes = int(numEgoNodes*pct/100)
    print('No. of selected ego-nodes: {}'.format(numSelectedEgoNodes))
    selectedPatterns = []
    for selectedEgoNode in egoNodeIds[:numSelectedEgoNodes]:
        selectedPatterns.append(selectedEgoNode+".*")

    print('Selected ego-nodes patterns: ')
    for pattern in selectedPatterns:
        print('\t' + pattern)

    edges = ["*.edges"]
    #Copy selected ego node files to dst
    #copy_files_with_patterns(src, dst, selectedPatterns)
    copy_files_with_patterns(src, dst, selectedPatterns)
