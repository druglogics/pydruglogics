pydruglogics.utils.BNetworkUtil
===============================

It provides file I/O, string parsing, Boolean network (BNet) conversions, and network interaction handling.

.. autofunction:: pydruglogics.utils.BNetworkUtil.BNetworkUtil.get_file_extension

    Get the file extension from a filename.


.. autofunction:: pydruglogics.utils.BNetworkUtil.BNetworkUtil.remove_extension

    Remove the extension from a filename and return the base name.


.. autofunction:: pydruglogics.utils.BNetworkUtil.BNetworkUtil.read_lines_from_file

    Read all lines from a file. Optionally skip empty lines and comments (lines starting with '#').


.. autofunction:: pydruglogics.utils.BNetworkUtil.BNetworkUtil.is_numeric_string

    Check if a value is a numeric string, int, or float.


.. autofunction:: pydruglogics.utils.BNetworkUtil.BNetworkUtil.parse_interaction

    Parse a network interaction string (e.g., from a SIF file) into a dictionary representation.


.. autofunction:: pydruglogics.utils.BNetworkUtil.create_interaction

    Create a blank interaction dictionary for a given target node.


.. autofunction:: pydruglogics.utils.BNetworkUtil.bnet_string_to_dict

    Convert a string in .bnet format into a Python dictionary.


.. autofunction:: pydruglogics.utils.BNetworkUtil.to_bnet_format

    Convert a list of Boolean equations to the '.bnet' string format.


.. autofunction:: pydruglogics.utils.BNetworkUtil.create_equation_from_bnet

    Parse a single line in .bnet format into the internal Boolean equation tuple.

