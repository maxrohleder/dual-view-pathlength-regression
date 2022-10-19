import xmltodict
import numpy as np


def spin_matrices_from_xml(path_to_projection_matrices):
    assert str(path_to_projection_matrices).endswith('.xml')
    with open(path_to_projection_matrices) as fd:
        contents = xmltodict.parse(fd.read())
        matrices = contents['hdr']['ElementList']['PROJECTION_MATRICES']
        proj_mat = np.zeros((len(matrices.keys()), 3, 4))
        for i, key in enumerate(matrices.keys()):
            value_string = matrices[key]
            proj_mat[i] = np.array(value_string.split(" "), order='C').reshape((3, 4))

    return proj_mat
