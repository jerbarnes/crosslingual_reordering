import re
import numpy as np

es_bwe = """     & Original 	 & 61.6 & 59.8 & 66.6 & 35.9 & 36.3 & 34.9 \\ 
		&& Reordered  & \bestproj{61.9} & 59.9 & 66.6 & 35.7 & \bestproj{36.5} & 34.9 \\ 
		&& N-ADJ  & 61.6 & \bestproj{60.2} & 66.6 & \bestproj{36.3} & \bestproj{36.5} & 34.9 \\ 
		&& Random  & 60.4 & 58.2 & 66.6 & 34.2 & 35.3 & 34.9 \\ 
		&& Only Lex.  & 59.1 & 26.2 & 53.0 & 29.8 & 16.4 & 30.7 \\ 
		&& No Lex.  & 57.9 & 55.7 & 63.4 & 33.1 & 33.7 & 33.3 \\ 
         """
ca_bwe = """     & Original & 68.2 & 65.7 & 68.2 & 31.9 & \bestproj{39.3} & 33.2 \\ 
		&& Reordered  & 68.4 & 65.5 & 68.2 & \bestproj{32.3} & 38.1 & 33.2 \\ 
		&& N-ADJ &  \bestproj{68.5} & \bestproj{65.8} & 68.2 & 31.8 & 38.9 & 33.2 \\ 
		&& Random & 66.9 & 64.4 & 68.2 & 31.7 & 36.4 & 33.2 \\ 
		&& Only Lex.  & 55.1 & 41.3 & 39.1 & 30.3 & 27.4 & 23.8 \\ 
		&& No Lex. & 64.8 & 63.0 & 63.1 & 30.4 & 35.9 & 31.2 \\ 
         """


es_mt = """	& Original & \bestmt{71.1} & \bestmt{67.4} & 69.6 & 47.1 & \bestmt{43.9} & 45.1 \\ 
		&& Random & 71.0 & 65.9 & 69.6 & \bestmt{48.0} & 41.7 & 45.1 \\ 
		&& Only Lexicon  & 61.8 & 47.3 & 54.5 & 32.2 & 27.3 & 35.7 \\ 
		&& No Lexicon & 63.2 & 59.1 & 65.4 & 42.9 & 37.9 & 42.0 \\ 
        """

ca_mt = """     & Original & \bestmt{79.0} & \bestmt{77.3} & 74.7 & \bestmt{53.1} & \bestmt{49.2} & 45.5 \\ 
		&& Random & 76.1 & 75.7 & 74.7 & 46.3 & 41.6 & 45.5 \\ 
		&& Only Lexicon  & 52.1 & 59.6 & 43.9 & 26.1 & 30.1 & 32.7 \\ 
		&& No Lexicon & 73.0 & 71.5 & 70.8 & 46.7 & 43.1 & 44.4 \\ 
        """

es_mono =  """  & Original &    \bestmono{73.9} & \bestmono{66.1} & 49.7 & \bestmono{43.2} & \bestmono{38.0} & 32.1\\
		&& Random &     71.7 & 60.4 & 49.7 & 41.7 & 33.6 & 32.1\\ 
		&& Only Lex.  & 47.2 & 46.5 & 45.2 & 26.0 & 25.6 & 27.0\\ 
		&& No Lex. &    68.2 & 65.2 & 47.9 & 40.1 & 37.2 & 30.3\\ 
           """

ca_mono = """   & Original &    \bestmono{77.0} & \bestmono{75.2} & 75.0 & \bestmono{50.3} & \bestmono{46.7} & 46.8\\
		&& Random &     73.1 & 67.9 & 75.0 & 46.5 & 40.3 & 46.8\\ 
		&& Only Lex.  & 43.0 & 56.7 & 39.6 & 13.9 & 32.2 & 16.7\\ 
		&& No Lex. &    73.6 & 75.4 & 74.8 & 48.3 & 45.9 & 45.8\\ 
          """


def clean(f, mod="bi"):
    """
    """
    raw = re.findall(r'[0-9]+\.[0-9]', f)
    if mod == "bi":
        array = raw[0:2]
    elif mod == "4class":
        array = raw[3:5]
    elif mod == "both":
        array = raw[0:2] + raw[3:5]
    else:
        array = raw
    return np.array(array, dtype=float)

def bwe_vs_mono():
    es_mono_orig = clean(es_mono.splitlines()[0], mod="all")
    es_bwe_orig = clean(es_bwe.splitlines()[0], mod="all")

    ca_mono_orig = clean(ca_mono.splitlines()[0], mod="all")
    ca_bwe_orig = clean(ca_bwe.splitlines()[0], mod="all")

    es_diff = es_mono_orig - es_bwe_orig
    ca_diff = ca_mono_orig - ca_bwe_orig

    return es_diff, ca_diff

def reordered():
    es_bwe_orig = clean(es_bwe.splitlines()[0], mod="both")
    es_bwe_reordered = clean(es_bwe.splitlines()[1], mod="both")

    ca_bwe_orig = clean(ca_bwe.splitlines()[0], mod="both")
    ca_bwe_reordered = clean(ca_bwe.splitlines()[1], mod="both")

    es_diff = es_bwe_reordered - es_bwe_orig
    ca_diff = ca_bwe_reordered - ca_bwe_orig

    return (es_diff + ca_diff) / 2
    
def nadj():
    es_bwe_orig = clean(es_bwe.splitlines()[0], mod="both")
    es_bwe_reordered = clean(es_bwe.splitlines()[2], mod="both")

    ca_bwe_orig = clean(ca_bwe.splitlines()[0], mod="both")
    ca_bwe_reordered = clean(ca_bwe.splitlines()[2], mod="both")

    es_diff = es_bwe_reordered - es_bwe_orig
    ca_diff = ca_bwe_reordered - ca_bwe_orig

    return (es_diff + ca_diff) / 2

def mono_rand():
    
    es_mono_orig = clean(es_mono.splitlines()[0], mod="both")
    es_mono_rand = clean(es_mono.splitlines()[1], mod="both")

    ca_mono_orig = clean(ca_mono.splitlines()[0], mod="both")
    ca_mono_rand = clean(ca_mono.splitlines()[1], mod="both")

    es_diff = es_mono_orig- es_mono_rand
    ca_diff = ca_mono_orig- ca_mono_rand

    return es_diff, ca_diff

def bwe_rand():
    
    es_mono_orig = clean(es_bwe.splitlines()[0], mod="both")
    es_mono_rand = clean(es_bwe.splitlines()[3], mod="both")

    ca_mono_orig = clean(ca_bwe.splitlines()[0], mod="both")
    ca_mono_rand = clean(ca_bwe.splitlines()[3], mod="both")

    es_diff = es_mono_orig- es_mono_rand
    ca_diff = ca_mono_orig- ca_mono_rand

    return es_diff, ca_diff

def mt_rand():
    
    es_mono_orig = clean(es_mt.splitlines()[0], mod="both")
    es_mono_rand = clean(es_mt.splitlines()[1], mod="both")

    ca_mono_orig = clean(ca_mt.splitlines()[0], mod="both")
    ca_mono_rand = clean(ca_mt.splitlines()[1], mod="both")

    es_diff = es_mono_orig- es_mono_rand
    ca_diff = ca_mono_orig- ca_mono_rand

    return es_diff, ca_diff
