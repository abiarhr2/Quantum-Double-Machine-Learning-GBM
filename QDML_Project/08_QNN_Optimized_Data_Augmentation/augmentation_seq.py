"""Reserved for the named SEQ method once its algorithm is specified."""

MISSING_SPECIFICATION = (
    'SEQ needs a paper or reference implementation, variable order, conditional '
    'models for binary and continuous variables, and sampling and missing-value rules.'
)


def generate(*args, **kwargs):
    raise NotImplementedError(MISSING_SPECIFICATION)
