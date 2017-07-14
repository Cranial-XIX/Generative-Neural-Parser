import constants
import torch

def signature(word, freq):
    """
    This routine returns a String that is the "signature" of the class of a
    word. For, example, it might represent whether it is a number of ends in -s. The
    strings returned by convention match the pattern UNK-.* , which is just assumed
    to not match any real word. Behavior depends on the unknownLevel (-uwm flag)
    passed in to the class. The recognized numbers are 1-5: 5 is fairly
    English-specific; 4, 3, and 2 look for various word features (digits, dashes,
    etc.) which are only vaguely English-specific; 1 uses the last two characters
    combined with a simple classification by capitalization.

    @param word
                The word to make a signature for
    @param freq
                The frequency of the word in training set
    @return
                A String that is its signature (equivalence class)
    """

    emb = torch.zeros(14)

    if freq >= constants.UNCOMMON_THRESHOLD:
        return emb, word

    sb = ["UNK"]

    # Reformed Mar 2004 (cdm); hopefully much better now.
    # { -CAPS, -INITC ap, -LC lowercase, 0 } +
    # { -KNOWNLC, 0 } + [only for INITC]
    # { -NUM, 0 } +
    # { -DASH, 0 } +
    # { -last lowered char(s) if known discriminating suffix, 0}
    wlen = len(word)
    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False

    for i in xrange(wlen):
        ch = word[i]
        if ch.isdigit():
            hasDigit = True
        elif ch == '-':
            hasDash = True
        elif ch.isalpha():
            if ch.islower():
                hasLower = True
            elif ch.istitle():
                hasLower = True
                numCaps += 1
            else:
                numCaps += 1

    ch0 = word[0]
    lowered = word.lower()
    if ch0.isupper() or ch0.istitle():
        sb.append("-CAPS")
        emb[0] = 1

    elif not ch0.isalpha() and numCaps > 0:
        sb.append("-CAPS")
        emb[0] = 1

    elif hasLower:
        sb.append("-LC")
        emb[1] = 1

    if hasDigit:
        sb.append("-NUM")
        emb[2] = 1

    if hasDash:
        sb.append("-DASH")
        emb[3] = 1
    if lowered.endswith('s') and wlen >= 3:
        # here length 3, so you don't miss out on ones like 80s
        ch2 = lowered[wlen - 2]

        # not -ess suffixes or greek/latin -us, -is
        if ch2 != 's' and ch2 != 'i' and ch2 != 'u':
            sb.append("-s")
            emb[4] = 1
    elif len(word) >= 5 and not hasDash and not (hasDigit and numCaps > 0):
        # don't do for very short words
        # Implement common discriminating suffixes
        if lowered.endswith("ed"):
            sb.append("-ed")
            emb[5] = 1
        elif lowered.endswith("ing"):
            sb.append("-ing")
            emb[6] = 1
        elif lowered.endswith("ion"):
            sb.append("-ion")
            emb[7] = 1
        elif lowered.endswith("er"):
            sb.append("-er")
            emb[8] = 1
        elif lowered.endswith("est"):
            sb.append("-est")
            emb[9] = 1
        elif lowered.endswith("ly"):
            sb.append("-ly")
            emb[10] = 1
        elif lowered.endswith("ity"):
            sb.append("-ity")
            emb[11] = 1
        elif lowered.endswith("y"):
            sb.append("-y")
            emb[12] = 1
        elif lowered.endswith("al"):
            sb.append("-al")
            emb[13] = 1

    return emb, ''.join(sb)
