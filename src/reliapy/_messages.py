def template_error():
    raise NotImplementedError('reliapy (Error 0) - this is a template.')


def type_error(var, type_var):
    raise TypeError('reliapy (Error 1) - the type of ' + var + ' must be: ' + type_var)


def shape_error(var, shape_var):
    raise ValueError('reliapy (Error 2) - the shape of ' + var + ' is not consistent.')


def not_implemented_error():
    raise NotImplementedError('reliapy (Error 4) - this option is not implemented.')

