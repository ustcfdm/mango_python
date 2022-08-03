import os
import re

def dir_reg(folder: str, reg: str, recursive: bool =False, ftype: str ='all'):
    '''
    List all pathnames in 'folder' that matches 'reg' in regular expression form.

    Parameters
    ----------
    folder : str
        The root folder where to find pathnames.
    reg : str
        Regular expression to match.
    recursive : bool, optional
        List pathnames recursively. The default is False.
    ftype : str, optional
        The path type to be matched ('dir', 'file', or 'all'). The default is 'all'.

    Raises
    ------
    ValueError
        An error when 'ftype' is not any of 'all', 'dir', or 'file'.

    Returns
    -------
    f : list
        The pathnames that matches the given regular expression.

    '''

    m = re.compile(reg)
    f = []

    for path, dirs, files in os.walk(folder):

        if ftype == 'all':
            ls = dirs + files
        elif ftype == 'dir':
            ls = dirs
        elif ftype == 'file':
            ls = files
        else:
            raise ValueError("Invalid ftype '{}'!".format(ftype))

        for item in ls:
            r = m.fullmatch(item)
            if r != None:
                f.append(os.path.join(path, r.string))

        if not recursive:
            break

    return f