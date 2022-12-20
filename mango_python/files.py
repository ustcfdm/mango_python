import os
import re
import shutil
import warnings
import pickle


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


def mkdir(path: str, overwrite: bool = False):
    """Create a directory with given options.

    Parameters
    ----------
    path : str
        Directory name
    overwrite : bool, optional
        Whether to overwrite the directory if it already exists, by default False
    """
    if os.path.isdir(path):
        if overwrite == False:
            # Do nothing
            return
        else:
            # Remove the path and create a new one
            
            warnings.warn(f'All the contents in "{path}" will be removed!', stacklevel=2)
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
    else:
        # path does not exist
        os.makedirs(path)
        

def load_pickle(filename: str) -> object:
    """Load object from file using pickle

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    object
        object
    """
    
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        
    return obj


def dump_pickle(filename: str, obj):
    """Dump object to file using pickle

    Parameters
    ----------
    filename : str
        filename
    obj : Any
        any variable
    """
    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)