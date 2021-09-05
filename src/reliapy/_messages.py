def template_error():
    """
    Error message for template functions.
    """
    raise NotImplementedError('reliapy (Error 0) - this is a template.')


def type_error(var, type_var):
    """
    Error message for variables with wrong data type.
    """
    raise TypeError('reliapy (Error 1) - the type of ' + var + ' must be: ' + type_var)


def shape_error(var):
    """
    Error message for variables with wrong shape.
    """
    raise ValueError('reliapy (Error 2) - the shape of ' + var + ' is not consistent.')


def not_implemented_error():
    """
    Error message for not implemented features.
    """
    raise NotImplementedError('reliapy (Error 4) - this option is not implemented.')


def value_error(var):
    """
    Error message for variables with wrong values.
    """
    raise ValueError('reliapy (Error 5) - the value of ' + var + ' is not acceptable.')

