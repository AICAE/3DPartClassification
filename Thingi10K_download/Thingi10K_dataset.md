

[![img](https://camo.githubusercontent.com/81660f0663d79e35760d49c46ca08a761a39bc4b3f209d243c30801207286e72/687474703a2f2f3364766973696f6e2e7072696e6365746f6e2e6564752f70726f6a656374732f323031342f4d6f64656c4e65742f7468756d626e61696c2e6a7067)](https://camo.githubusercontent.com/81660f0663d79e35760d49c46ca08a761a39bc4b3f209d243c30801207286e72/687474703a2f2f3364766973696f6e2e7072696e6365746f6e2e6564752f70726f6a656374732f323031342f4d6f64656c4e65742f7468756d626e61696c2e6a7067)

**Thingi10K: A Dataset of 10,000 3D-Printing Models (2016)** [[Link\]](https://ten-thousand-models.appspot.com/)
10,000 models from featured “things” on thingiverse.com, suitable for testing 3D printing techniques such as structural analysis , shape optimization, or solid geometry operations.

9GB  stl raw meshes

Total number of files: 10,000

STL files: 9956
OBJ files: 42
PLY files: 1
OFF files: 1



Our query interface is very useful in dissecting the dataset based on mesh quality measures. For example, all single-component, manifold solid meshes without self-intersection and degeneracies can be obtained with the query term 

“num component=1, is manifold, is solid, without self-intersection, without degeneracy”. 

A python download script will be available,  but no meta data can be download



https://docs.scrapy.org/en/latest/topics/dynamic-content.html



metadata: 

| #Vertices                  | :    | 82     |
| -------------------------- | ---- | ------ |
| #Faces                     | :    | 160    |
| #Components                | :    | 1      |
| Euler                      | :    | 2      |
| Genus                      | :    | 0      |
| Closed                     | :    | True   |
| Oriented                   | :    | True   |
| Self-intersecting          | :    | False  |
| Vertex manifold            | :    | True   |
| Edge manifold              | :    | True   |
| Combinatorially degenerate | :    | False  |
| Geometrically degenerate   | :    | False  |
| Duplicated faces           | :    | False  |
| PWN                        | :    | True   |
| Solid                      | :    | True   |
| Total area                 | :    | 5964.6 |



