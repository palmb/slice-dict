#!/usr/bin/env python

import pandas as pd
from goodbadugly import Frame

if __name__ == '__main__':
    f = Frame(
        a=pd.Series(range(2)),
        aVeryLongNameThatIsNowFine=pd.Series(range(4)),
        # b=pd.DataFrame(columns=list('abc'), index=range(7)),
        x=pd.Series(dtype=float)
    )
    pd.DataFrame.index
    print(f)
