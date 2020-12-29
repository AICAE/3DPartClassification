/// copyright,  qingfeng xia, personal work, not belong to UKAEA
/// Created on Sunday, Nov 01 2020

#include "occ_project.h"

typedef Eigen::MatrixXf Image;
const std::vector<std::string> PNAMES = {"_XY", "_YZ", "_ZX"};  // rolling
const std::string IM_SUFFIX = ".csv";


//typedef std::array<double, 3> Coordinate;   gp_Pnt,  gp_XYZ instead
void save_image(const Image& im, const std::string& filename)
{
    const static Eigen::IOFormat CSVFormat(4, 0, ", ", "\n");
    std::ofstream file(filename.c_str());
    file << im.format(CSVFormat);
}


std::set<std::pair<size_t, size_t>> calc_thickness(const IntersectionMat& mat, Image& im)
{
    std::set<std::pair<size_t, size_t>> error_pos;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();
            if(n==0)
                im(r, c) = 0.0f;
            else if (n%2 == 1)
            {
                std::cout << "Error: intersection vector size " << n << " is not an even number " << r << ", " << c <<"\n";
                im(r, c) = 0.0f;   // set zero, or intoperation later
                error_pos.insert({r, c});
            }
            else if(n==2)
                im(r, c) = std::abs((*p)[0] - (*p)[1]);
            else
            {
                // sorted then get the diff for every 2 
                std::vector<double> v(*p);
                std::sort(v.begin(), v.end());
                double t = 0.0;
                for(size_t i = 0; i<n%2; i++ )
                {
                    t +=  v[i*2+1] - v[i*2+1];
                }
                im(r, c) = std::abs(t);
            }
        }
    }
    return error_pos;
}

/// if single pixel missing, it is acceptable for interpolation
/// if there is large area of missing pixels, interpolation has bad quality.
void interoplate_thickness(const std::set<std::pair<size_t, size_t>> error_pos, Image& im)
{
    for(const auto& it: error_pos)
    {
        size_t r = it.first;
        size_t c = it.second;
        std::vector<float> values;
        for (size_t i=1; r+i < im.rows(); i++)
        {
            if( error_pos.find({r+i, c}) == error_pos.end() )
            {
                values.push_back(im(r+i, c));
                break; 
            }
        }
        for (size_t i=1; r-i >= 0; i++)
        {
            if( error_pos.find({r-i, c}) == error_pos.end())
            {
                values.push_back(im(r-i, c));
                break; 
            }
        }
        for (size_t i=1; c+i < im.cols(); i++)
        {
            if( error_pos.find({r, c+i}) == error_pos.end() )
            {
                values.push_back(im(r, c+i));
                break; 
            }
        }
        for (size_t i=1; c-i >= 0; i++)
        {
            if( error_pos.find({r, c-i}) == error_pos.end() )
            {
                values.push_back(im(r, c-i));
                break; 
            }
        }
        assert(std::size(values) <= 4);
        im(r, c) = std::accumulate(values.cbegin(), values.cend(), 0.0) / std::size(values);
    }
}


void test_IndexedMap()
{
    // is key in set ordered? will insertion change index or previous?
    NCollection_IndexedDataMap<const char*, int> map;
    auto i1 = map.Add("B", 1);
    auto i2 = map.Add("A", 2);
    std::cout << "NCollection_IndexedDataMap index: " << i1 << i2 << std::endl;
}


/// find after faceting, 
void save_data(const IntersectionData& data, const std::string& output_file_stem)
{
    // 4. find the line-face intersection
    for(int i=0; i<NVIEWS; i++)
    {
        const auto& mat = *data[i];
        Image im(mat.rows(), mat.cols());
        auto error_pos = calc_thickness(mat, im);
        double threshold = 0.05;
        if (std::size(error_pos) > mat.rows() * mat.cols() * threshold)
        {
            std::cout << "Error: image will not be saved as total number of odd thickness vector size " << 
                std::size(error_pos) << " is higher than the threshold percentage " <<  0.05 * 100 <<"\n";
        }
        else
        {
            if(std::size(error_pos) > 0)
            {
                std::cout << "Warning: total number of odd thickness vector size " << std::size(error_pos) << 
                    ",  less than threshold" <<  0.05 * 100 << " percentage, will be interpolated\n";
                interoplate_thickness(error_pos, im);
            }
            save_image(im, output_file_stem + PNAMES[i] + IM_SUFFIX);
        }
    }
}


/// slow but robust: find the intersection line segments by boolean operation
double intersect_bop(const TopoDS_Shape& shape, const TopoDS_Edge& edge)
{
    double t = 0;

    BRepAlgoAPI_Common bc{shape, edge};
    //bc.SetRunParallel(occInternalParallel);
    // bc.SetFuzzyValue(); tolerance?
    bc.SetNonDestructive(Standard_True);
    if (not bc.IsDone())
    {
        std::cout << "BRepAlgoAPI_Common is not done\n";
    }

    std::vector<TopoDS_Shape> res; // output is limited to Solid shape type
    for (TopExp_Explorer anExp(bc.Shape(), TopAbs_EDGE); anExp.More(); anExp.Next())
    {
        GProp_GProps l_props;
        BRepGProp::LinearProperties(anExp.Current(), l_props, true);
        t +=  l_props.Mass();
    }

    return t;
}

/// this BOP method is very slow but precise
///  and it uses a grid different from triangulation method
int bop(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    double min[DIM], max[DIM], space[DIM];
    box.Get(min[0], min[1], min[2], max[0], max[1], max[2]);

    for (size_t i=0; i< DIM; i++)
        space[i] = (max[i] - min[i])/NGRIDS[i];

    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        size_t c_index = (1 + iaxis)%3;
        size_t z_index = (iaxis + 2)%3;
        //gp_Vec dir(0, 0, 0);
        //dir.SetCoord(z_index + 1, 1.0);

        Image im(NGRIDS[r_index], NGRIDS[c_index]);

        TopoDS_Builder cBuilder;
        TopoDS_Compound merged;
        cBuilder.MakeCompound(merged);

        for (size_t r=0; r< im.rows(); r++)
        {
            double r_value = min[r_index] + space[r_index] * (r + 0.5);
            for (size_t c=0; c< im.cols(); c++)
            {
                double c_value = min[c_index] + space[c_index] * (c + 0.5);
                // create the curve, then surface, IntCS
                gp_Pnt p1; 
                p1.SetCoord(r_index+1, r_value);
                p1.SetCoord(c_index+1, c_value); 
                p1.SetCoord(z_index+1, min[z_index]);
                gp_Pnt p2 = p1;
                p2.SetCoord(z_index+1, max[z_index]);

                auto e = BRepBuilderAPI_MakeEdge(p1, p2).Edge();
                im(r, c) = intersect_bop(shape, e);
                cBuilder.Add(merged, e);
            }
        }

        save_image(im, output_file_stem + "_BOP" + PNAMES[iaxis] + IM_SUFFIX);
#ifndef NDEBUG
        BRepTools::Write(merged, (output_file_stem + PNAMES[iaxis] + "_grid.brep").c_str());
#endif
    }
    // BRepTools::Write(shape, (output_file_stem + "_shape.brep").c_str());
    return 0;
}


/// triangulation, fast less precise than BOP
int project(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    MeshData mesh;
    mesh.grid_info = generate_grid(shape);

    IntersectionData data;
    init_intersection_data(data);
    auto faces = get_free_faces(shape);
    double linDefl = 0.25; // or 0.1 times of pixel grid space?
    for(const auto& f: *faces)
    {
        mesh.triangles = generate_mesh(f, mesh.local_transform, linDefl);
        calc_intersections(mesh.triangles, mesh.grid_info, mesh.local_transform, data);
    }
    // save thickness matrix as numpy array,  scale it?  save as image?

    save_data(data, output_file_stem);

#ifndef NDEBUG
    auto stl_exporter = StlAPI_Writer(); // high level API
    stl_exporter.Write(shape, (input + "_meshed.stl").c_str()); 
    // shape must has mesh for each face
#endif

    return 0;
}




int project_mesh(std::string input, const std::string& output_file_stem, const BoundBoxType bbox)
{

    IntersectionData data;
    init_intersection_data(data);

    MeshData mesh;
    mesh.grid_info = generate_grid(bbox);
    mesh.local_transform = gp_Trsf();  // not necessary
    mesh.triangles = read_mesh(input);
    calc_intersections(mesh.triangles, mesh.grid_info, mesh.local_transform, data);

    // save thickness matrix as numpy array,  scale it?  save as image?
    save_data(data, output_file_stem);

    return 0;
}


#include <argparse.hpp>

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("program name");

  program.add_argument("input")
    .required()
    .help("input geometry file path");

  program.add_argument("-o", "--output")
    .help("specify the output file stem");

  program.add_argument("--grid").nargs(3)
    .help("image pixel for x, y, z as an integer array")
    .default_value(std::vector<int>{64, 64, 64})
    .action([](const std::string& value) { return std::stoi(value); });


  program.add_argument("--bbox").nargs(6)
    .help("boundbox values for stl off input file: xmin, ymin, zmin, xmax, ymax, zmax")
    //.default_value(std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
    .action([](const std::string& value) { return std::stod(value); });

  program.add_argument("--bop")
    .help("use the very slow BOP method")
    .default_value(false)
    .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }
  
    //test_IndexedMap();
    auto input = program.get<std::string>("input");
    std::string output_stem = input;
    if(program.present("-o"))
    {
        output_stem = program.get<std::string>("-o");
    }

    // only if found ".stl", ".off"  must be converted to stl file
    if(input.find(".stl") !=std::string::npos)
    {
        BoundBoxType bbox;
        if (true)  // program.present("--bbox")
        {
            bbox = program.get<BoundBoxType>("--bbox");
            project_mesh(input, output_stem, bbox);
        }
    }
    else
    {
        if(program.get<bool>("--bop"))
        {
            bop(input, output_stem);  // working but extremely slow, can be used to compare speed
        }
        else
        {
            project(input, output_stem);
        }
    }
    
    return 0;
}