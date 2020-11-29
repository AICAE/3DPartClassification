/// copyright,  qingfeng xia, personal work, not belong to UKAEA
/// Created on Sunday, Nov 01 2020

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <algorithm>    // std::sort
#include <stdexcept>
#include <system_error>
#include <optional>

#include "Standard_Version.hxx"

#include "gp_Lin.hxx"
#include "Bnd_OBB.hxx"
#include "BRepBndLib.hxx"
#include "BRep_Tool.hxx"
#include "BRepTools.hxx"

#include "BRepMesh_IncrementalMesh.hxx"
#include "Poly_Array1OfTriangle.hxx"
#include "TColgp_Array1OfPnt.hxx"
//#include "TNCOLStd_Array1OfInteger.hxx"
#include "Poly_Triangulation.hxx"
#include "Poly_PolygonOnTriangulation.hxx"

#include "TopoDS.hxx"
#include "TopoDS_Shape.hxx"
#include "TopoDS_Face.hxx"

#include "TopTools_HSequenceOfShape.hxx"
#include "TopTools_IndexedMapOfShape.hxx"
#include "TopExp_Explorer.hxx"
#include "TopExp.hxx"

#include "TopLoc_Location.hxx"
#include "BRep_Builder.hxx"
#include "BRepBuilderAPI_Transform.hxx"
#include "BRepBuilderAPI_MakeEdge.hxx"

#include "GProp_GProps.hxx"
#include "BRepGProp.hxx"
#include "BRepAlgoAPI_Common.hxx"

#include "OSD_Path.hxx"
#include "StlAPI_Reader.hxx"
#if OCC_VERSION_HEX >= 0x070300
#include "RWStl.hxx"  // OCCT 7.4, read Shape from STL file
#endif
#include "StlAPI_Writer.hxx"


#include <Eigen/Dense>

const unsigned int NVIEWS = 3;

std::array<size_t, 3> NGRIDS = {64, 64, 64};  // make it diff to test image size
const std::vector<std::string> PNAMES = {"_XY", "_YZ", "_ZX"};  // rolling
const std::string IM_SUFFIX = ".csv";

typedef gp_Vec Coordinate;

template <typename T> 
using Mat = Eigen::Matrix<std::shared_ptr<T>, Eigen::Dynamic, Eigen::Dynamic> ;

typedef std::array<std::shared_ptr<Mat<std::vector<double>>>, NVIEWS> IntersectionData;
typedef std::vector<double> BoundBoxType;

template <typename T> 
void init_matrix(Mat<T>& mat)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            mat(r, c) = std::make_shared<T>();
        }
    }
}

IntersectionData init_intersection_data()
{
    IntersectionData data;
    data[0] = std::make_shared<Mat<std::vector<double>>>(NGRIDS[0], NGRIDS[1]);
    data[1] = std::make_shared<Mat<std::vector<double>>>(NGRIDS[1], NGRIDS[2]);
    data[2] = std::make_shared<Mat<std::vector<double>>>(NGRIDS[2], NGRIDS[0]);

    for(int i=0; i<NVIEWS; i++)
    {
        Mat<std::vector<double>>& mat = (*data[i]);
        init_matrix(mat);
    }
    return data;
}



typedef float scalar;  // single-precision float can be push to GPU for acceleration
typedef Eigen::MatrixXf Image;
//typedef std::array<double, 3> Coordinate;   gp_Pnt,  gp_XYZ instead
void save_image(const Image& im, const std::string& filename)
{
    const static Eigen::IOFormat CSVFormat(4, 0, ", ", "\n");
    std::ofstream file(filename.c_str());
    file << im.format(CSVFormat);
}

std::shared_ptr<std::vector<TopoDS_Face>> get_external_faces(const TopoDS_Shape &shape) 
{

    auto externalFaces = std::make_shared<std::vector<TopoDS_Face>>();

    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        const TopoDS_Face &face = TopoDS::Face(ex.Current());
        // check if it is an external face, skip interior face

        // Important note: For the surface map, face equivalence is defined
        // by TopoDS_Shape::IsSame(), which ignores the orientation.
        //if (face.Orientation() == TopAbs_EXTERNAL)
            externalFaces->push_back(face);
    }
    return externalFaces;
}

const int DIM=3;  

/// linear grid (equally-spaced in one axis), 3D grid, voxel
struct GridInfo
{
    std::array<double, DIM> min;
    //std::array<double, DIM> max;
    std::array<double, DIM> spaces;
    std::array<size_t, DIM> nsteps;  // npixels (cells) in each axis
};


GridInfo generate_grid(const TopoDS_Shape& shape)
{
    // must be rotated with AABB = OBB,  AABB (Axis-align) is `class Bnd_Box`

    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    double xmin, xmax, ymin, ymax, zmin, zmax;
    box.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    /// use slightly bigger space, to make sure full edge are zero thickness (void)
    std::array<double, DIM> spaces = {(xmax-xmin)/(NGRIDS[0]-1), (ymax-ymin)/(NGRIDS[1]-1), (zmax-zmin)/(NGRIDS[2]-1)};
    std::array<double, DIM> starts = {xmin - spaces[0] * 0.5, ymin - spaces[1] * 0.5, zmin - spaces[2] * 0.5};  // first cell's center
      
    GridInfo gInfo{starts, spaces, NGRIDS};

    return gInfo;
}


/// http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
/// https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
bool in_triangle(double xx, double yy, double x[3], double y[3])
{
    double  denominator = ((y[1] - y[2])*(x[0] - x[2]) + (x[2] - x[1])*(y[0] - y[2]));
    double  a = ((y[1] - y[2])*(xx - x[2]) + (x[2]- x[1])*(yy - y[2])) / denominator;
    double  b = ((y[2] - y[0])*(xx - x[2]) + (x[0] - x[2])*(yy - y[2])) / denominator;
    double  c = 1 - a - b;
    
    return 0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1;
}

/// calc of normal depends on triangle vertices seq
double area(const std::vector<const Coordinate*> tri)
{
    gp_Vec ab = (*tri[1]) - (*tri[0]);
    gp_Vec ac = (*tri[2]) - (*tri[0]);
    gp_Vec n = ab.Crossed(ac);      // cross product
    // divided by 2 is the triangle area
}

/// Möller and Trumbore, « Fast, Minimum Storage Ray-Triangle Intersection », Journal of Graphics Tools, vol. 2,‎ 1997, p. 21–28
/// it is more efficient on GPU
/// http://geomalgorithms.com/a06-_intersect-2.html
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 =  disjoint (no intersect)
//             1 =  intersect in unique point I1
//             2 =  are in the same plane
int calc_intersection(const std::vector<const Coordinate*> tri, const std::vector<const Coordinate*> R, gp_Vec& I)
{
    gp_Vec    u, v, n;              // triangle vectors
    gp_Vec    w0, w;                // ray vectors
    double     r, a, b;              // params to calc ray-plane intersect
    const double SMALL_NUM = 1e-8;

    // get triangle edge vectors and plane normal
    u = (*tri[1]) - (*tri[0]);
    v = (*tri[2]) - (*tri[0]);
    n = u.Crossed(v);                 // cross product
    if (n.Magnitude() < 10e-12)    // triangle is degenerate, zero area?
        return -1;                     // do not deal with this case
         
    w0 = *R[0] - (*tri[0]);
    gp_Vec dir = *R[1] - *R[0];
    a = -n.Dot(w0);
    b = n.Dot(dir);           // ray direction vector
    if (std::fabs(b) < SMALL_NUM) {     // ray is  parallel to triangle plane
        if (a == 0)                 // ray lies in triangle plane, co-planar
            return 2;
        else 
            return 0;              // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)                    // ray goes away from triangle
        return 0;                   // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    I = *R[0] + r * dir;            // intersect point of ray and plane

    float    uu, uv, vv, wu, wv, D;
    uu = u.Dot(u);
    uv = u.Dot(v);
    vv = v.Dot(v);
    w = I - (*tri[0]);
    wu = w.Dot(u);
    wv = w.Dot(v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)         // I is outside T
        return 0;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return 0;

    return 1;
}


bool has_intersection(const std::vector<const Coordinate*> poly, double vi, double vj, int iaxis)
{
    double x[3], y[3];
    for(int i = 0; i < 3; i++)
    {
        x[i] = (*poly[i]).Coord(iaxis+1);
        int iy = iaxis + 2 > 3? 1:iaxis + 2;
        y[i] = (*poly[i]).Coord(iy);
    }
    
    return in_triangle(vi, vj, x, y);
}


std::vector<std::array<size_t,2>> get_index_ranges(const std::vector<const Coordinate*> polygon, const GridInfo ginfo)
{
    std::vector<std::array<size_t,2>> ranges;
    for(int iaxis = 0; iaxis < DIM; iaxis++)
    {
        std::vector<double> v;
        for(const auto& p: polygon)
        {
            v.push_back((*p).Coord(iaxis+1));   /// NOTE: index for X is 1 not 0
        }
        auto min_i = std::distance(v.cbegin(), std::min_element(v.cbegin(), v.cend()));
        size_t imin = size_t((v[min_i] - ginfo.min[iaxis]) / ginfo.spaces[iaxis] ) - 1;
        if(imin <0)
        {
            imin = 0;
        }
        auto max_i = std::distance(v.cbegin(), std::max_element(v.cbegin(), v.cend()));
        /// should be rounded up for max, not to miss some point
        size_t imax = size_t((v[max_i] - ginfo.min[iaxis]) / ginfo.spaces[iaxis] ) + 1;
        if (imax > ginfo.nsteps[iaxis])
        {
            std::cout << "Error:  i_upper " << imax << " > grid number " << ginfo.nsteps[iaxis] << std::endl;
            imax = ginfo.nsteps[iaxis];
        }
        ranges.push_back({imin, imax});
    }
    return ranges;
}

void insert_intersections(const std::vector<Coordinate>& points, const std::array<int, 3>& tri, const GridInfo& ginfo, IntersectionData& data)
{
    const std::vector<const Coordinate*> triangle = {points.data() + tri[0],  points.data() + tri[1],  points.data() + tri[2]};
    // get box bound for the tri, project to one plane, get possible indices from the grid
    const auto iranges = get_index_ranges(triangle, ginfo);
    // if line and triangle intersection, then save the intersection coord's third-component
    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        const auto [r_start, r_end] = iranges[r_index];
        size_t c_index = (1 + iaxis)==3? 0: 1 + iaxis;
        const auto [c_start, c_end] = iranges[c_index];
        size_t third_index = (iaxis + 2)%3;
        gp_Vec dir(0, 0, 0);
        dir.SetCoord(third_index + 1, 1.0);
        double third_min = ginfo.min[third_index] - ginfo.spaces[third_index];
        double third_max = ginfo.min[third_index] + ginfo.spaces[third_index] * (ginfo.nsteps[third_index]+1);

        Mat<std::vector<double>>& mat = *data[iaxis];
        for(size_t r = r_start; r < r_end; r++)
        {
            for(size_t c = c_start; c<c_end; c++)
            {
                double r_value = ginfo.min[r_index] + ginfo.spaces[r_index] * r;
                double c_value = ginfo.min[c_index] + ginfo.spaces[c_index] * c;

                gp_Vec pos(0, 0, 0);
                pos.SetCoord(r_index+1, r_value);
                pos.SetCoord(c_index+1, c_value);
                pos.SetCoord(third_index + 1, third_min);
                gp_Vec p2 = pos;
                pos.SetCoord(third_index + 1, third_max);
                //gp_Lin l(pos, dir);

                #if 0
                auto ret = has_intersection(triangle, r_value, c_value, iaxis);
                if(ret)
                {
                    gp_Vec I(0, 0, 0);
                    int m = calc_intersection(triangle, l, I);
                    //if (m == 1)
                    mat(r, c)->push_back(I.Coord(third_index + 1));
                }
                #else
                gp_Vec I(0, 0, 0);
                int m = calc_intersection(triangle, {&pos, &p2}, I);
                if (m == 1)
                {
                    auto  h = I.Coord(third_index + 1);
                    if (h < third_max && h > third_min)
                        mat(r, c)->push_back(h);
                }
                #endif
            }
        }
    }

}


void calc_intersections(const Handle(Poly_Triangulation) triangles, const GridInfo& ginfo, const gp_Trsf local_transform, IntersectionData& data) 
{

    const TColgp_Array1OfPnt &nodes = triangles->Nodes();
    // retrieve triangle coordinates
    std::vector<Coordinate> points;   // NCollection_Array1<>
    points.reserve(nodes.Upper());
    if (nodes.Lower() == 1 )  // one started array and index
    {
        points.push_back({INFINITY, INFINITY, INFINITY});  // if we use index 0, then all inf
    }
    for (int i = nodes.Lower(); i <= nodes.Upper(); i++) {
        Standard_Real x, y, z;
        nodes(i).Coord(x, y, z);
        local_transform.Transforms(x, y, z);
        points.push_back({x,y,z});
    }

    std::array<int, 3> indices;
    const Poly_Array1OfTriangle &tris = triangles->Triangles();
    //     std::cout << "Face has " << tris.Length() << " triangles" << std::endl;
    for (int i = tris.Lower(); i <= tris.Upper(); i++) 
    {
        // get the node indexes for this triangle
        const Poly_Triangle &tri = tris(i);
        tri.Get(indices[0], indices[1], indices[2]);
        insert_intersections(points, indices, ginfo, data);
    }

}

void calc_thickness(const Mat<std::vector<double>>& mat, Image& im)
{
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
                im(r, c) = 0.0f;   // set zero, or intoperation
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
}

void check_thickness(const Mat<std::vector<double>>& mat, int iaxis)
{
    // build points and mesh to debug
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();
            if (n%2 == 1)
            {

            }
        }
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

struct mesh_data
{
    Handle(Poly_Triangulation) triangles;
    GridInfo grid_info;
    gp_Trsf local_transform;
};

/// find after faceting, 
void save_data(const IntersectionData& data, const std::string& output_file_stem)
{
    // 4. find the line-face intersection
    for(int i=0; i<NVIEWS; i++)
    {
        const auto& mat = *data[i];
        Image im(mat.rows(), mat.cols());
        calc_thickness(mat, im);
        save_image(im, output_file_stem + PNAMES[i] + IM_SUFFIX);
    }
}


const Handle(Poly_Triangulation)  generate_mesh(TopoDS_Face f, gp_Trsf& local_transform )
{
    double tolerance = 0.25;  // 5 times of pixel grid space?
    // mesh each face, by given best tols
    BRepMesh_IncrementalMesh facets(f, tolerance, true, 5);

    TopLoc_Location local_loc;
    // BRep_Tool::Triangulation is for face only
    const Handle(Poly_Triangulation) triangles  = BRep_Tool::Triangulation(f, local_loc);
    // also get local transform
    local_transform = local_loc;
    // why TopoLoc_Location  can be assigned to gp_Trsf?

    if (triangles.IsNull()) {
        std::cout << "No facets for surface" << std::endl;
    }

    return triangles;
}

/// stp, brep files
TopoDS_Shape read_geometry(std::string filename)
{
    // 2. image resolution, setup grid
    TopoDS_Shape shape;
    BRep_Builder builder;
    BRepTools::Read(shape, filename.c_str(), builder);
    return shape;
}


/// STEP, BREP shape, first of all, imprint and merge
TopoDS_Shape  prepare_shape(std::string input)
{
    // 1. rotate view according to boundbox
    TopoDS_Shape shape = read_geometry(input);

    Bnd_OBB obb;   // YHSize  H means half length
    BRepBndLib::AddOBB(shape, obb);
    if(!obb.IsAABox())
    {
        gp_Ax3 obb_ax(obb.Center(), obb.ZDirection(), obb.XDirection());
        gp_Trsf trsf;
        gp_Ax3 orig;
        orig.SetLocation(obb.Center());
        trsf.SetTransformation(obb_ax, gp_Ax3());                            
        auto t = BRepBuilderAPI_Transform(shape, trsf, true);
        //shape = t.Shape();
    }
    return shape;
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

int bop(std::string input, const std::string& output_file_stem)
{
    auto shape = prepare_shape(input);
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

    auto shape = prepare_shape(input);
    mesh_data mesh;
    mesh.grid_info = generate_grid(shape);

    IntersectionData data = init_intersection_data();
    auto faces = get_external_faces(shape);
    for(const auto& f: *faces)
    {
        mesh.triangles = generate_mesh(f, mesh.local_transform);
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


/// OFF STL surface mesh, OCCT 7.4
Handle(Poly_Triangulation)  read_mesh(std::string filename)
{
    #if OCC_VERSION_HEX >= 0x070300
    auto t = RWStl::ReadFile(filename.c_str());
    if(t)
        return t;
    else
        throw std::runtime_error(std::string(" can not find triangle from STL file"));    
    #else
    throw std::runtime_error(std::string("occ 7.4 is needed to read stl mesh as Poly_Triangulation"));
    #endif
}


GridInfo generate_grid(const BoundBoxType bbox)
{
    double xmin, ymin, zmin, xmax, ymax, zmax;
    xmin = bbox[0], ymin = bbox[1], zmin = bbox[2];
    xmax = bbox[3], ymax = bbox[4], zmax = bbox[5];
    int sh = 3;
    std::array<double, DIM> spaces = {(xmax-xmin)/(NGRIDS[0]-sh), 
            (ymax-ymin)/(NGRIDS[1]-sh), (zmax-zmin)/(NGRIDS[2]-sh)};
    std::array<double, DIM> starts = {xmin - spaces[0] * 0.5 * sh, 
            ymin - spaces[1] * 0.5 * sh, zmin - spaces[2] * 0.5 * sh};  // first cell's center
      
    GridInfo gInfo{starts, spaces, NGRIDS};

    return gInfo;
}

int project_mesh(std::string input, const std::string& output_file_stem, const BoundBoxType bbox)
{

    IntersectionData data = init_intersection_data();

    mesh_data mesh;
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

  //program.add_argument("-o", "--output")
  //  .help("specify the output file.");

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
            project(input, output_stem);
    }
    
    return 0;
}