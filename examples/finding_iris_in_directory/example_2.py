from irisloc import find_irises_in_dir, make_collage

#filenames should contain information about center of image
find_irises_in_dir("./test_images", blur=True, bri_contr=False)
make_collage("./output", (7, 6))