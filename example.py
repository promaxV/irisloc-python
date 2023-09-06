from irisloc import find_iris_by_path, get_center_on_image

#click on pupil center and press ESC
center = get_center_on_image("./eye.png")

iris = find_iris_by_path("./eye.png", "./eye_with_iris.png", center, mode='iris')
print("That is what 'iris' mode returns:", iris)
pupil = find_iris_by_path("./eye.png", "./eye_with_pupil.png", center, mode='pupil')
print("That is what 'pupil' mode returns:", pupil)
both = find_iris_by_path("./eye.png", "./eye_with_both.png", center, mode='both')
print("That is what 'both' mode returns:", both)