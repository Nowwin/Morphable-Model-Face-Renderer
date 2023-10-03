import numpy as np
import h5py

def check_winding_order(v0, v1, v2):
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_product = np.cross(edge1, edge2)
    
    if cross_product[2] > 0:
        return "Counter-Clockwise"
    elif cross_product[2] < 0:
        return "Clockwise"
    else:
        return "Collinear"

def fix_winding_order(faces):
    # This function assumes `faces` is a 2D numpy array where each column is a face
    # and the rows are the indices of the vertices in that face.
    fixed_faces = faces.copy()
    fixed_faces[[1, 2], :] = fixed_faces[[2, 1], :]  # Swap the second and third vertices
    return fixed_faces

def compute_face_normals(vertices, faces):
    face_normals = np.zeros((faces.shape[1], 3))
    for i in range(faces.shape[1]):
        face = faces[:, i]
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        face_normals[i] = normal / np.linalg.norm(normal)  # Normalize the normal
    return face_normals

def compute_vertex_normals(vertices, faces, face_normals):
    vertex_normals = np.zeros_like(vertices)
    face_count = np.zeros(vertices.shape[0]) 

    for i in range(faces.shape[1]):
        face = faces[:, i]
        for vertex in face:
            vertex_normals[vertex] += face_normals[i]
            face_count[vertex] += 1

    # Average and normalize
    vertex_normals /= face_count[:, np.newaxis]
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    return vertex_normals


def load_data(filename):
    with h5py.File(filename, 'r') as f:
        mean_shape_dataset = f['shape/model/mean'][:]
        mean_shape_pca_dataset = f['shape/model/pcaBasis'][:]
        mean_texture_dataset = f['color/model/mean'][:]
        mean_texture_pca_dataset = f['color/model/pcaBasis'][:]
        cells_dataset = f['shape/representer/cells'][:]
        points_dataset = f['shape/representer/points'][:]
    
    return mean_shape_dataset, mean_shape_pca_dataset, mean_texture_dataset, mean_texture_pca_dataset, cells_dataset

def generateNewFace(mean_shape, mean_shape_pca, mean_texture, mean_texture_pca):
    #For now it's doing nothing
    return mean_shape, mean_texture

def normalizeData(data):
    # Find the minimum and maximum values for each axis
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Find the maximum range among all axes
    max_range = np.max(max_vals - min_vals)
    
    # Calculate the scaling factor
    scale_factor = 2 / max_range
    
    # Translate the data to have a minimum of 0
    translated_data = data - min_vals
    
    # Apply the scaling factor, and then translate to the range [-1, 1]
    normalized_data = translated_data * scale_factor - 1
    
    return normalized_data

def writeObjFile(vertices, normal, faces):
    with open('model.obj', 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

        for vertexNormal in normal:
            obj_file.write(f'vn {vertexNormal[0]} {vertexNormal[1]} {vertexNormal[2]}\n')
        
        for i in range(faces.shape[1]):
            face = faces[:, i]  # Get the i-th face
            # OBJ indices are 1-based, so add 1 to each index
            obj_file.write(f'f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n')



def main():
    filename = '../imageData/model2019_fullHead.h5'
    mean_shape, mean_shape_pca, mean_texture, mean_texture_pca, cells = load_data(filename)
    print(mean_shape.shape)
    print(cells.shape)
    print(cells[0:3, 0:2])

    if(check_winding_order(np.array([mean_shape[cells[0, 0]*3], mean_shape[cells[0, 0]*3 + 1], mean_shape[cells[0, 0]*3 + 2]]),
                        np.array([mean_shape[cells[1, 0]*3], mean_shape[cells[1, 0]*3 + 1], mean_shape[cells[1, 0]*3 + 2]]),
                        np.array([mean_shape[cells[2, 0]*3], mean_shape[cells[2, 0]*3 + 1], mean_shape[cells[2, 0]*3 + 2]])) 
                        == "Clockwise") :
                        print("Correcting the winding order\n")
                        cells = fix_winding_order(cells)

    final_shape, final_texture = generateNewFace(mean_shape, mean_shape_pca, mean_texture, mean_texture_pca)

    #Making it n, 3 so that it's easier to process
    reshaped_vertices = final_shape.reshape(-1, 3)

    final_normal = compute_vertex_normals(reshaped_vertices, cells, compute_face_normals(reshaped_vertices, cells))
    final_coordinates = normalizeData(reshaped_vertices)

    writeObjFile(final_coordinates, final_normal, cells)




    

if __name__ == "__main__":
    main()
