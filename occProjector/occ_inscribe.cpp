#include "occ_project.h"

#include <argparse.hpp>


int bop(std::string input, const std::string& output_file_stem)
{
    auto shape = prepare_shape(input);
    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    // BOP OBB subtract thisShape

    // get the fragment with the max volume if not single, may also worth know the second largest

    // use other methods to calc inscribed shape

    return 0;

}

/// lower and upper height
typedef Mat<std::pair<scalar, scalar>> Region;
typedef std::vector<scalar> InscribedShapeType;

/// sphere, box, bnd_box
InscribedShapeType get_inscribed_shape(const Region& mat)
{


}

/// calc real volume if grid_cell's area is given as the second parameter
scalar volume(const Region& mat, const scalar cell_area = 1)
{
    scalar v = 0;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if(p)
                v += std::abs((*p).second - (*p).first);
        }
    }
    return v * cell_area;
}


/// see also calc_thickness()
void getAllVoidConvexRegions(const IntersectionMat& mat, std::vector<Region>& regions)
{
    std::set<std::pair<size_t, size_t>> error_pos;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();

        }
    }
}

InscribedShapeType calc_inscribe_shape(const IntersectionData& data, const GridInfo& grid)
{

    // select only one axis, hmax, hmin,  float_aproximate
    int iaxis = 0;

    std::vector<Region> regions;
    getAllVoidConvexRegions(*(data[iaxis]), regions);
    std::vector<scalar> vols;
    for(const auto& r: regions)
    {
        vols.push_back(volume(r));
    }
    size_t imax = std::distance(vols.cbegin(), std::max_element(vols.cbegin(), vols.cend()));

    return get_inscribed_shape(regions[imax]);
}

/// triangulation, fast less precise than BOP
int inscribe(std::string input, const std::string& output_file_stem)
{
    auto shape = prepare_shape(input);
    MeshData mesh;
    mesh.grid_info = generate_grid(shape);

    IntersectionData data;
    init_intersection_data(data);
    auto faces = get_free_faces(shape);
    for(const auto& f: *faces)
    {
        mesh.triangles = generate_mesh(f, mesh.local_transform);
        calc_intersections(mesh.triangles, mesh.grid_info, mesh.local_transform, data);
    }
    // grid may not detect tiny void. 
    calc_inscribe_shape(data, mesh.grid_info);

    /// BOP check, there is not union volume
    return 0;

}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("occ_inscribe");

  program.add_argument("input")
    .required()
    .help("input geometry file path");

  program.add_argument("-o", "--output")
    .help("specify the output file stem");

  program.add_argument("--grid").nargs(3)
    .help("image pixel for x, y, z as an integer array")
    .default_value(std::vector<int>{64, 64, 64})
    .action([](const std::string& value) { return std::stoi(value); });

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
    std::string output_stem = program.get<std::string>("-o");

        if(program.get<bool>("--bop"))
        {
            bop(input, output_stem);  // working but extremely slow, can be used to compare speed
        }
        else
        {
            inscribe(input, output_stem);
        }
    
    
    return 0;
}