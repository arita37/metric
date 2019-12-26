# -*- coding: utf-8 -*-
import ast
import fnmatch
import inspect
import math
import operator
import os
import re
import sys
from builtins import int, open, range, str, zip
from collections import OrderedDict
from importlib import import_module
from pkgutil import walk_packages
import regex
import pkgutil
import shutil
import stat


###################################################################################################
SEP =  "\\"  if  "win" in sys.platform  else "/"


COLS_NAME = [
    "module_name",
    "module_version",
    "full_name",
    "prefix",
    "obj_name",
    "obj_doc",
    "object_type",
    "arg_full",
]
NAN = float("nan")


def zdoc():
    source = inspect.getsource(ztest)
    print(source)


##################################################################################################
def str_join(*members):
    """join_memebers(member1, ...) takes an orbitrary number of
    arguments of type 'str'and concatinates them with using the '.' separator"""
    return ".".join(members)


def np_merge(*dicts):
    container = {}
    for d in dicts:
        container.update(d)
    return container


def module_load(name_or_path="") :
    name_or_path = os.path.abspath(name_or_path)

    try :
        module = import_module(m_name)
        print("Module imported", module)
        return module
    except :
        #sys.path.add(m_name)  ## Absolute path
        mpath = name_or_path.split(SEP)[:-1]
        mpath = SEP.join( mpath )
        print(mpath)
        sys.path.insert(0, mpath )  ## Absolute path
        m_name = name_or_path.split(SEP)[-1]
        module = import_module(m_name)
        print("Module imported", module, flush=True)
        return module



def module_getname(name) :
   if SEP not in name : return name 
   else  :
    return name.split(SEP)[-1]



def module_getpath(name) :
    return name

##################################################################################################
class Module:
    """class Module(module_name)
    Class Module gets all the submodules, classes, class_methods and functions of 
    the module taken as an argument by its instance. Every instance of the class has:
    ATTRIBUTES: Module.module_name, self.module, self.submodules, self.functions, self.classes, self.class_methods
    METHODS: self.get_module_version(self), self.get_submodules(self), self.get_functions(self), self.get_classes(self),
    self.get_class_methods(), self.isimported(self, name), self.get_mlattr(self, full_name), self.get_submodule(self, attr)."""

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = module_load(self.module_name)
        self.module_isbuiltin = self.get_module_isbuiltin()
        self.submodules = self.get_submodules()
        self.functions = self.get_functions()
        self.functions_built = self.get_builtin_functions()
        self.classes = self.get_classes()
        self.class_methods = self.get_class_methods()

    def get_module_version(self):
        """get_module_version(self) Module method
        return the version of the module taken as an instance argument."""
        try :
            return self.module.__version__
        except :
            return 0

    def get_module_isbuiltin(self):
        builtin_modules = list(sys.builtin_module_names)
        for x in builtin_modules:
            if self.module.__name__ == x:
                return True
        return False 

    def get_submodules(self):
        """get_submodules(self) Module method
        return a list of submodules of the module taken as an instance argument."""
        submodules = {} 
        
        try:      
            for loader, name, is_pkg in walk_packages(self.module.__path__, self.module.__name__ + "."):
                if self.is_imported(name):
                    submodules[name] = self.get_submodule(self.get_mlattr(name))
        except:
#            for loader, name, is_pkg in walk_packages(sys.path, self.module.__name__ + "."):
            for loader, name, is_pkg in walk_packages(None, self.module.__name__ + "."):
                if self.is_imported(name):
                    submodules[name] = self.get_submodule(self.get_mlattr(name))
                
        return submodules
    
    def get_functions(self):
        """get_functions(self) Module method
        return a list of functions of the module taken as an instance argument."""
        functions = {}
        for submodule_name, submodule in self.submodules.items():
            for function_name, function in inspect.getmembers(
                submodule, lambda f: inspect.isfunction(f) or inspect.isbuiltin(f) 
            ):
                functions[str_join(submodule_name, function_name)] = function
        return functions

    def get_builtin_functions(self):
        """get_builtin_functions(self) Module method
        return a list of functions of the module taken as an instance argument."""
        functions_built = {}
        if self.module_isbuiltin:
            mod = eval(self.module.__name__)
        else:
            mod = self.module.__name__
        for function_name, function in inspect.getmembers(
            mod, lambda f: inspect.isfunction(f) or inspect.isbuiltin(f) 
            ):                
                functions_built[str_join(self.module.__name__, function_name)] = function
        return functions_built

    def get_classes(self):
        """get_classes(self) Module method
        return a list of classes of the module taken as an instance argument."""
        classes = {}
        for submodule_name, submodule in self.submodules.items():
            for class_name, class_ in inspect.getmembers(submodule, lambda c: inspect.isclass(c)):
                classes[str_join(submodule_name, class_name)] = class_
        return classes

    def get_class_methods(self):
        """get_class_methods(self) Module method
        return a list of class methods of the module taken as an instance argument."""
        methods = {}
        for class_name, class_ in self.classes.items():
            for method_name, method in inspect.getmembers(
                class_, lambda m: inspect.ismethod(m) or inspect.isbuiltin(m)
            ):
                methods[str_join(class_name, method_name)] = method
        return methods

    def is_imported(self, submodule_name):
        """is_imported(self, submodule_name) Module method
        retrun True if submodule was imported and False otherwise."""
        return submodule_name in sys.modules

    def get_mlattr(self, full_name):
        """get_mlattr(self, full_name) Module method
        return a multi-level attribute of an object."""
        return full_name.split(".", 1)[1]

    def get_submodule(self, attr):
        """get_submodule(self, attr) Module method
        return submodule object of the module by its attribute."""
        return operator.attrgetter(attr)(self.module)


def obj_get_name(obj):
    """get_name(obj) return object name."""
    return obj.__name__


def obj_get_doc_string(obj):
    """get_doc_string(obj) return object doc string"""
    return re.sub("\x08.", "", pydoc.render_doc(obj)) or obj.__doc__


def obj_get_prefix(name):
    """get_prefix(name) return object prefix."""
    return name.split(".", 1)[1].rsplit(".", 1)[0]


def str_strip_text(string):
    """str_strip_text(string) strip \b and \n literals off the string."""
    return re.sub("\x08.", "", string.replace("\n", ""))


def obj_get_signature(obj):
    obj_name = obj.__name__
    obj_doc = str_strip_text(pydoc.render_doc(obj))
    match = regex.findall(obj_name + "(\((?>[^()]+|(?1))*\))", obj_doc)[:2]
    if match:
        if len(match) > 1:
            signature = (
                match[0][1:-1] if match[0][1:-1] != "..." and match[0] != "" else match[1][1:-1]
            )
            return signature
        else:
            return match[0][1:-1] if match[0][1:-1] != "..." else ""
    else:
        return ""

def obj_get_full_signature(obj):
    arg_full = OrderedDict()
    try:
        args = inspect.signature(obj)
    except:
        args = ""
    arguments = str(args)
    arguments = re.sub('()', '', arguments)
    arg_full[1] = arguments
    return arg_full

    
def obj_get_args(obj):
    arguments = OrderedDict()
    if inspect.isbuiltin(obj):
        obj_signature = obj_get_signature(obj)
        if obj_signature:
            pattern = "\w+=[-+]?[0-9]*\.?[0-9]+|\w+=\w+|\w+=\[.+?\]|\w+=\(.+?\)|[\w=']+"
            items = re.findall(pattern, obj_signature)
            for item in items:
                split_item = item.split("=")
                if len(split_item) == 2:
                    arguments[split_item[0]] = split_item[1]
                elif len(split_item) == 1:
                    arguments[split_item[0]] = NAN
            return arguments
        else:
            return {}
    else:
        argspec = inspect.getfullargspec(obj)
        args = argspec.args
        defaults = argspec.defaults
        if defaults:
            args_with_default_values = OrderedDict(zip(args[-len(defaults) :], defaults))
            for arg in args:
                if arg in args_with_default_values:
                    arguments[arg] = args_with_default_values[arg]
                else:
                    arguments[arg] = NAN
            return arguments
        else:
            return OrderedDict(zip(args, [NAN] * len(args)))


def obj_guess_arg_type(arg_default_values):
    types = []
    for arg_value in arg_default_values:
        if isinstance(arg_value, str):
            try:
                types.append(type(ast.literal_eval(arg_value)).__name__)
            except ValueError:
                types.append("str")
            except SyntaxError:
                types.append(NAN)
        elif isinstance(arg_value, float) and math.isnan(arg_value):
            types.append(NAN)
        else:
            types.append(type(arg_value).__name__)
    return tuple(types)


def obj_get_arginfo(obj, args):
    """get_arginfo(obj, args) return a tuple of the object argument info."""
    return ("arg_info",) * len(args)


def obj_get_nametype(obj):
    """get_name(obj) return object name."""
    types = {"function": inspect.isfunction, "method": inspect.ismethod, "class": inspect.isclass}
    for obj_type, inspect_type in types.items():
        if inspect_type(obj):
            return obj_type
    return None


def obj_class_ispecial(obj):
    try:
        inspect.getfullargspec(obj.__init__)
    except TypeError:
        return False
    else:
        if inspect.isclass(obj):
            return True
        else:
            return False


def obj_get_type(x):
    # eval
    if isinstance(x, str):
        return "str"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"


#############################################################################################################
def module_signature_get(module_name):
    """module_signature(module_name) return a dictionary containing information
       about the module functions and methods"""
    module = Module(module_name)
    if module.module_isbuiltin:
        members = module.functions_built
    else:
        members = np_merge(module.functions, module.classes, module.class_methods)
    
    
    doc_df = {
        "module_name": module_name,
        "module_version": module.get_module_version(),
        "full_name": [],
        "prefix": [],
        "obj_name": [],
        "obj_doc": [],
        ## TODO:   add function_type column
        # 'obj_type'    class / class.method /  function / decorator ....
        #"function_type":[],
        "object_type": [],
        "arg_full": [],
        "arg": [],
        "arg_default_value": [],
        "arg_type": [],
        "arg_info": [],
        
    }

    for member_name, member in members.items():
        
        isclass = obj_class_ispecial(member)
        isfunction = inspect.isfunction(member)
        ismethod = inspect.ismethod(member)
                        
        if isclass or isfunction or ismethod or module.module_isbuiltin:
            doc_df["full_name"].append(member_name)
            doc_df["prefix"].append(obj_get_prefix(member_name))
            doc_df["obj_name"].append(obj_get_name(member))
            doc_df["obj_doc"].append(obj_get_doc_string(member))
            doc_df["object_type"].append(obj_get_nametype(member))
            doc_df["arg"].append(tuple(obj_get_args(member.__init__ if isclass else member).keys()))
            doc_df["arg_default_value"].append(
                tuple(obj_get_args(member.__init__ if isclass else member).values())
            )
            doc_df["arg_type"].append(obj_guess_arg_type(doc_df["arg_default_value"][-1]))
            doc_df["arg_info"].append(obj_get_arginfo(member, doc_df["arg"][-1]))
            if not module.module_isbuiltin:
                doc_df["arg_full"].append(tuple(obj_get_full_signature(member.__init__ if isclass else member).values()))
            else:
                doc_df["arg_full"].append(None)
                   
    return doc_df


def module_signature_write(module_name, outputfile="", return_df=0, isdebug=0):
    """  Write down the files.
         
    """
    df = module_signature_get(module_name)
    df = pd_df_format(pd.DataFrame(df), COLS_NAME)
    df = df.sort_values("full_name", ascending=True)
    
    if return_df == 1:
        return df  # return df
    else:
        outputfile = (
            outputfile
            if outputfile != ""
            else os.path.join(os.getcwd(), str_join("doc_" + module_name, "csv"))
        )
        if isdebug:
            print("Signature Writing")
        print(outputfile)
        df.to_csv(outputfile, index=False, mode="w")




######################################################################################################
############## Code Search #################################################################################
def conda_path_get(subfolder="package/F:/"):
    if os.__file__.find("envs") > -1:
        DIRANA = os.__file__.split("envs")[0] + "/"  # Anaconda from linux
    else:
        DIRANA = os.__file__.split("Lib")[0] + "/"  # Anaconda from root

    os_name = sys.platform[:3]
    if subfolder == "package":
        DIR2 = None
        if os_name == "lin":
            DIR2 = DIRANA + "/Lib/site-packages/"
        elif os_name == "win":
            DIR2 = DIRANA + "/Lib/site-packages/"
        return DIR2


def os_file_listall(dir1, pattern="*.*", dirlevel=1, onlyfolder=0):
    """ dirpath, filename, fullpath
   # DIRCWD=r"D:\_devs\Python01\project"
   # aa= listallfile(DIRCWD, "*.*", 2)
   # aa[0][30];   aa[2][30]
  """
    matches = {}
    dir1 = dir1.rstrip(os.path.sep)
    num_sep = dir1.count(os.path.sep)
    matches["dirpath"] = []
    matches["filename"] = []
    matches["fullpath"] = []

    for root, dirs, files in os.walk(dir1):
        num_sep_this = root.count(os.path.sep)
        if num_sep + dirlevel <= num_sep_this:
            del dirs[:]
        for f in fnmatch.filter(files, pattern):
            matches["dirpath"].append(os.path.splitext(f)[0])
            matches["filename"].append(os.path.splitext(f)[1])
            matches["fullpath"].append(os.path.join(root, f))
    return matches



################################################################################################
def np_list_dropduplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def code_parse_line(li, pattern_type="import/import_externa"):
    """
    External Packages
  """
    ### Import pattern
    if pattern_type == "import":
        if li.find("from") > -1:
            l = li[li.find("from") + 4 : li.find("import")].strip().split(",")
        else:
            l = li.strip().split("import ")[1].strip().split(",")

        l = [x for x in l if x != ""]
        l = np_list_dropduplicate(l)
        return l

    # Only external
    if pattern_type == "import_extern":
        if li.find("from") > -1:
            l = li[li.find("from") + 4 : li.find("import")].strip().split(",")
        else:
            l = li.strip().split("import ")[1].strip().split(",")

        l = [x for x in l if x != ""]
        l = [x for x in l if x[0] != "."]
        l = [x.split(".")[0].split("as")[0].split("#")[0].strip() for x in l]
        l = np_list_dropduplicate(l)
        return l




######################################################################################################
IIX = 0


def log(*args, reset=0):
    global IIX
    IIX = IIX + 1
    a = ",".join(args)
    print( f"\n--{IIX} : {a}", flush=True)



