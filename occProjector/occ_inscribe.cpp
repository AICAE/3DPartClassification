#include "occ_project.h"

#include <argparse.hpp>


int bop(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    // BOP OBB subtract thisShape

    // get the fragment with the max volume if not single, may also worth know the second largest

    // use other methods to calc inscribed shape

    return 0;

}

/// lower and upper height
typedef std::pair<scalar, scalar> Range;
const Range VOID_RANGE = Range{0, 0};
typedef Eigen::Matrix<Range, Eigen::Dynamic, Eigen::Dynamic> Region;
typedef std::vector<scalar> InscribedShapeType;

/// calc all shape types: sphere, box, bnd_box, return the max volume
InscribedShapeType generate_inscribed_shape(const Region& mat)
{
    // sphere is easiest, point cloud

}

InscribedShapeType calc_inscribed_sphere(const Region& mat)
{
    // sphere is easiest, point cloud

}

InscribedShapeType calc_inscribed_obb(const Region& mat)
{
    // calc the principle axis first, then rotate.

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

            v += std::abs(p.second - p.first);
        }
    }
    return v * cell_area;
}

inline bool approximate(scalar a, scalar b)
{
    return std::abs(a-b) < 1e-3  or std::abs(a-b) < std::abs(a+b) * 1e-4;  // CONSIDER
}

void sort_intersection(IntersectionMat& mat)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if (p)
                std::sort((*p).begin(), (*p).end());           
        }
    }
}

/// check only 2 directions, is that sufficient?
/// todo: there is possibility that more than one ranges are connected with previous range
/// consider  n=2   n=2 has been considered outside this function
/// so extend_region has been achieved
Range get_connected_range(const std::vector<scalar>& v, const Region& u, size_t r, size_t c,
        const std::pair<scalar, scalar> minmax)
{
    auto n = v.size();
    std::vector<Range> prevs;
    if ( c > 0)
    {
        prevs.push_back( u(r, c-1) );
    }
    if ( r > 0)
    {
        prevs.push_back(u(r-1, c));
    }
    for(const auto& prev: prevs)
    {
        // todo: may calc the max range, instead of return the first
        if(prev.first < v[0])
        {
            return {minmax.first, v[0]};
        }
        if(prev.second > v[n-1])
        {
            return {v[n-1], minmax.second};
        }
        for(size_t i = 1; i<v.size()-1; i+=2)
        {
            if(prev.first < v[i+1] or prev.second > v[i])
                return {v[i], v[i+1]};
        }
    }
    return VOID_RANGE;  // not connected with 
}

/// n_length: height vector size must be >= 4
void calc_hollow_region(const IntersectionMat& mat, int n_length, const std::pair<scalar, scalar> minmax, 
    Region& u, std::vector<Region>& regions)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if (p)
            {
                auto n = p->size();
                if (n == n_length)
                {
                    u(r,c) = {(*p)[1], (*p)[2]};
                }
                // this is extend_region()
                else if (n > 0)  // todo: create a new region for n+2?
                {
                    u(r,c) = get_connected_range(*p, u, r, c, minmax); // compare with r-1 and c-1, then 
                }
                else  // as bbox thickness
                {
                    {/* code */}
                }
                
            }
            else
            {
                u(r,c) = minmax;
            }
            
        }
    }
}

/// not needed
void extend_region(const IntersectionMat& mat, Region& r_n)
{

}

/// todo: a curve of VOID_RANGE splits a region into multiple isolated regions
/// image segmentation algorithm
void segment_region(const Region& r, std::vector<Region>& result_regions)
{
    result_regions.push_back(r);  /// todo: assuming it is single region
}

/// see also calc_thickness(), IntersectionMat must has been sorted before call this
void get_all_void_regions(const IntersectionMat& mat, const std::pair<scalar, scalar> minmax, 
    std::vector<Region>& regions)
{
    Region u(mat.rows(), mat.cols()); // upper void region,  each item will be assigned later
    Region l(mat.rows(), mat.cols()); // lower void region,
    Region h(mat.rows(), mat.cols()); // hollow void,   set default value to zero in loop later

    std::set<std::pair<size_t, size_t>> error_pos;
    size_t n_max=0;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if (p)
            {
                const auto n = p->size();
                if (n == 2)
                {
                    if(approximate((*p)[0], minmax.first))
                        l(r,c) = VOID_RANGE;
                    else
                    {
                        l(r,c) = {minmax.first, (*p)[0]};
                    }
                    if(approximate((*p)[n-1], minmax.second))
                        u(r,c) = VOID_RANGE;   // void, zero thickness
                    else
                    {
                        u(r,c) = {(*p)[n-1], minmax.second};
                    }
                }
                else if (n % 2)  // non even height vector, this is kind of error
                {
                    error_pos.insert({r, c});
                }
                if (n > n_max)
                    n_max = n;
            }
            else
            {
                u(r,c) = minmax;
                l(r,c) = minmax;
            }
        }
    }
    regions.emplace_back(l);
    regions.emplace_back(u);

    const int n_max_limit = 4;     // todo, higher level is not supported
    calc_hollow_region(mat, 4, minmax, h, regions);
    //extend_region(mat, h);
    //segment_region(h);
    regions.emplace_back(h);
}


InscribedShapeType calc_inscribe_shape(const IntersectionData& data, const BoundBoxType bbox, int iaxis = 0)
{

    // select only one axis, hmax, hmin,  float_aproximate
    std::pair<scalar, scalar> minmax = {bbox[iaxis], bbox[DIM+iaxis]};

    std::vector<Region> regions;
    auto mat = *(data[iaxis]);
    sort_intersection(mat);
    get_all_void_regions(mat, minmax, regions);
    std::vector<scalar> vols;
    for(const auto& r: regions)
    {
        vols.push_back(volume(r));
    }
    size_t imax = std::distance(vols.cbegin(), std::max_element(vols.cbegin(), vols.cend()));

    return generate_inscribed_shape(regions[imax]);
}

/// triangulation, fast less precise than BOP
int inscribe(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    MeshData mesh;
    mesh.grid_info = generate_grid(shape);

    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    BoundBoxType bbox(DIM*2);
    box.Get(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

    IntersectionData data;
    init_intersection_data(data);
    auto faces = get_free_faces(shape);
    for(const auto& f: *faces)
    {
        mesh.triangles = generate_mesh(f, mesh.local_transform);
        calc_intersections(mesh.triangles, mesh.grid_info, mesh.local_transform, data);
    }
    // grid may not detect tiny void. 
    calc_inscribe_shape(data, bbox);

    /// translate back to shape OBB gp_Ax3, then save

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
    std::string output_stem = input;
    if(program.present("-o"))
    {
        output_stem = program.get<std::string>("-o");
    }

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