#include "occ_project.h"

/// lower and upper height
typedef std::pair<scalar, scalar> Range;
const Range VOID_RANGE = Range{0, 0};
typedef Eigen::Matrix<Range, Eigen::Dynamic, Eigen::Dynamic> Region;
typedef std::vector<scalar> InscribedShapeT;


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

/// using the thickness as the weight, divided by volume
/// image pixel coordinate (r, c) has not yet translated into 3D spatical coordinate
std::array<scalar, DIM> center_of_mass(const Region& mat, const scalar volume)
{
    size_t n = 0;
    scalar w = 0;
    scalar x,y,z;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c < mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if (p != VOID_RANGE)
            {
                w = std::abs(p.second - p.first);
                x += w * r;
                y += w * c;
                z += w * (p.first + w * 0.5); // needs reconsider, too high
                n += 1;
            }
        }
    }
    return {x/volume, y/volume, z/volume} ;
}

/// this translation apply to XY projection only!
Coordinate translate(std::array<scalar, DIM> coord, const GridInfo& gInfo)
{
    int r = 0;  int c = 1; int t = 2;
    auto x = gInfo.min[r] + gInfo.spaces[r] * coord[0];
    auto y = gInfo.min[c] + gInfo.spaces[c] * coord[1];
    return Coordinate(x, y, coord[2]);
}

/// this translation apply to XY projection only!
std::array<size_t, DIM> calc_grid_indices(std::array<scalar, DIM> coord, const GridInfo& gInfo)
{
    std::array<size_t, DIM> ind;
    for(size_t i=0; i<DIM; i++)
    {
        ind[i] = std::round((coord[i] - gInfo.min[i])/ gInfo.spaces[i]);
        //ind[i] = std::clamp(ind[i], 0, gInfo.nsteps[i]);
    }
    return ind;
}

//void shrink(Bnd_Sphere& s, const Coordinate& c)


/// if region is convex, center of mass is the center of inscribed sphere
/// the init sphere is big, then needs shrink by check all other nearby region
InscribedShapeT init_inscribed_sphere(const Region& mat)
{
    // sphere is easiest, point cloud
    auto v = volume(mat);
    auto c = center_of_mass(mat, v);
    auto range = mat(std::round(c[0]), std::round(c[1]));
    // bug, range == VOID_SPACE
    auto R = (range.second - range.first) * 0.5;
    auto h = (range.second + range.first) * 0.5;
    return {c[0], c[1], h, R};
}

/// todo: optimization 
InscribedShapeT calc_inscribed_sphere(const Region& mat, const GridInfo& gInfo)
{
    InscribedShapeT s = init_inscribed_sphere(mat);
    auto xy = translate({s[0], s[1], s[2]}, gInfo);
    return {xy.X(), xy.Y(), s[2], s[3]};
}

/// based on cylinder
InscribedShapeT calc_inscribed_obb(const Region& mat)
{
    // calc the principle axis first, then rotate.
}

/// BrepBndLib
InscribedShapeT calc_principal_axis(const Region& mat)
{
    // calc the principle axis first, then rotate.

}

/// may use linear algebra deal with this optimization problem
/// the target is the max volume
InscribedShapeT calc_inscribed_cylinder(const Region& mat)
{
    // calc the principle axis first, then rotate.

}


/// calc all shape types: sphere, box, bnd_box, return the max volume
InscribedShapeT generate_inscribed_shape(const Region& mat, const BoundBoxType& bbox)
{
    const GridInfo gInfo = generate_grid(bbox);
    return calc_inscribed_sphere(mat, gInfo);
}

inline bool approximate(scalar a, scalar b)
{
    return std::abs(a-b) < 1e-3  or std::abs(a-b) < std::abs(a+b) * 1e-4;  // CONSIDER
}

/// this is not tested
void sort_intersection(IntersectionMat& mat)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            auto& p = mat(r,c);
            if (p and p->size())
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
    std::vector<Range> nb;
    if ( c > 0)  // out of range check
        nb.push_back( u(r, c-1) );
    if ( r > 0)
        nb.push_back(u(r-1, c));
    if ( c < u.cols() -1 )
        nb.push_back(u(r+1, c));
    if (r < u.rows() -1 )
        nb.push_back(u(r, c+1) );

    for(const auto& prev: nb)
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
    return VOID_RANGE;  // not connected with other region
}

/// 
void extend_region(const IntersectionMat& mat, int n_length, const std::pair<scalar, scalar> minmax, 
    Region& u, std::vector<Region>& regions)
{
    std::vector<scalar> full_range = {minmax.first, minmax.second};
    for (size_t r = 0; r < mat.rows(); r++)
    {
        for (size_t c = 0; c < mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            if (p)
            {
                auto n = p->size();

                if (n != n_length && n != 0)  // bug here: n can be zero!
                {
                    u(r,c) = get_connected_range(*p, u, r, c, minmax); 
                }               
            }
            else
            {
                //u(r,c) = minmax;  // may also check surrounding 4 directions
                u(r,c) = get_connected_range(full_range, u, r, c, minmax); 
            }
        }
    }
}

/// n_length: height vector size must be >= 4
/// WARNING: currenlty, only support n=4
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
            }    
        }
    }
    extend_region(mat,  n_length,  minmax, u, regions);
}


/// save the thickness image to CSV so to plot
void save_region(const Region& rg, const std::string& filename)
{
    std::ofstream file(filename.c_str());
    for (size_t r=0; r< rg.rows(); r++)
    {
        for (size_t c=0; c< rg.cols()-1; c++)
        {
            const auto& p = rg(r, c);
            file << std::abs(p.second - p.first) << ", ";
        }
        const auto& p = rg(r, rg.cols()-1);
        file << std::abs(p.second - p.first);
        file << std::endl;
    }
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
            const auto n = p->size();
            if (n>0)
            {
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


InscribedShapeT calc_inscribe_shape(const IntersectionData& data, const BoundBoxType bbox, int iaxis = 0)
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
    // debug file output
    save_region(regions[imax], "max_region.csv");
    // python3 -c "import numpy as np; import matplotlib.pyplot as plt; m = np.loadtxt("max_region.csv", delimiter=','); plt.imshow(m); plt.show()"

    return generate_inscribed_shape(regions[imax], bbox);
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
    auto s = calc_inscribe_shape(data, bbox);
    std::cout << "inscribed shape: ";
    for (int x: s)
    std::cout << x << std::endl;

    /// translate s back to shape OBB gp_Ax3, then save
    gp_Ax3 obb_ax(obb.Center(), obb.ZDirection(), obb.XDirection());
    gp_Trsf trsf;
    trsf.SetTransformation(gp::XOY(), obb_ax);
    // todo:   



    /// BOP check, there is not union volume
    return 0;

}