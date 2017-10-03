def signature(token, known, knownlc=False):
    if known:
        return token
    elif len(token) == 0:
        return 'UNK'
    else:
        numCaps = 0
        hasDigit = False
        hasDash = False
        hasLower = False
        for char in token:
            if char.isdigit():
                hasDigit = True
            elif char == '-':
                hasDash = True
            elif char.isalpha():
                if char.islower():
                    hasLower = True
                elif char.isupper():
                    numCaps += 1
        result = 'UNK'
        lower = token.lower()
        ch0 = token[0]
        if ch0.isupper():
            if numCaps == 1:
                result = result + '-INITC'    
                if knownlc:
                    result = result + '-KNOWNLC'
            else:
                result = result + '-CAPS'
        elif not(ch0.isalpha()) and numCaps > 0:
            result = result + '-CAPS'
        elif hasLower:
            result = result + '-LC'
        if hasDigit:
            result = result + '-NUM'
        if hasDash:
            result = result + '-DASH' 
        if lower[-1] == 's' and len(lower) >= 3:
            ch2 = lower[-2]
            if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                result = result + '-s'
        elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
            if lower[-2:] == 'ed':
                result = result + '-ed'
            elif lower[-3:] == 'ing':
                result = result + '-ing'
            elif lower[-3:] == 'ion':
                result = result + '-ion'
            elif lower[-2:] == 'er':
                result = result + '-er'            
            elif lower[-3:] == 'est':
                result = result + '-est'
            elif lower[-2:] == 'ly':
                result = result + '-ly'
            elif lower[-3:] == 'ity':
                result = result + '-ity'
            elif lower[-1] == 'y':
                result = result + '-y'
            elif lower[-2:] == 'al':
                result = result + '-al'
        return result