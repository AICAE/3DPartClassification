#pragma once
/// copyright,  qingfeng xia, personal work, not belong to UKAEA
/// Created on Sunday, Nov 01 2020
/// split into occ_project.h, Dec 23 2020
/// Although inscribe just needs 1 view, in order to reuse the code, 3 view are created

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <unordered_map>
#include <algorithm>    // std::sort
#include <stdexcept>
#include <system_error>
#include <optional>

#include "gp_Vec.hxx"
#include "Poly_Triangulation.hxx"

#include "TopoDS.hxx"
#include "TopoDS_Shape.hxx"
#include "TopoDS_Face.hxx"


// these below may be moved into cpp file
#include "Standard_Version.hxx"

#include "gp_Lin.hxx"
#include "Bnd_OBB.hxx"
#include "BRepBndLib.hxx"
#include "BRep_Tool.hxx"
#include "BRepTools.hxx"

#include "BRepAlgoAPI_Fuse.hxx"
#include "BRepAlgoAPI_Common.hxx"

#include "STEPControl_Reader.hxx"
#include "IFSelect_ReturnStatus.hxx"

#include "BRepMesh_IncrementalMesh.hxx"
#include "Poly_Array1OfTriangle.hxx"
#include "TColgp_Array1OfPnt.hxx"
//#include "TNCOLStd_Array1OfInteger.hxx"
#include "Poly_PolygonOnTriangulation.hxx"

#include "TopTools_HSequenceOfShape.hxx"
#include "TopTools_IndexedMapOfShape.hxx"
#include "TopExp_Explorer.hxx"
#include "TopExp.hxx"

#include "TopLoc_Location.hxx"
#include "BRep_Builder.hxx"
#include "BRepBuilderAPI_Transform.hxx"
#include "BRepBuilderAPI_MakeEdge.hxx"
#include "BRepBuilderAPI_Copy.hxx"

#include "GProp_GProps.hxx"
#include "BRepGProp.hxx"

#include "OSD_Path.hxx"
#include "OSD_Exception.hxx"
#include "StlAPI_Reader.hxx"
#if OCC_VERSION_HEX >= 0x070300
#include "RWStl.hxx"  // OCCT 7.4, read Shape from STL file
#endif
#include "StlAPI_Writer.hxx"

typedef gp_Vec Coordinate;

#include <Eigen/Dense>
typedef double scalar;  // single-precision float can be push to GPU for acceleration

#ifdef BUILD_INSCRIBE
const unsigned int NVIEWS = 3;
#else
const unsigned int NVIEWS = 3;
#endif
const int DIM=3;

// this can be set, but must be set before call any functions.
std::array<size_t, DIM> NGRIDS = {64, 64, 64};  // make it diff to test image size

// normalized XYZ according to OBB, if input geometry has been oriented, then use AABB, not OBB
bool USE_OBB = false;
bool USE_CUBE_BOX = true;  // xyz use the same and max thickness (longest boundbox length)
const bool NORMALIZED_THICKNESS = true;  // normalized to [0,1] according to boundbox minmax

template <typename T> 
using Mat = Eigen::Matrix<std::shared_ptr<T>, Eigen::Dynamic, Eigen::Dynamic> ;

/// shared_pointer to empty vector instead of nullptr
typedef Mat<std::vector<scalar>> IntersectionMat;
typedef std::array<std::shared_ptr<IntersectionMat>, NVIEWS> IntersectionData;
typedef std::vector<scalar> BoundBoxType;

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


/// linear point grid (equally-spaced in one axis), 3D grid, slightly bigger than bound box
struct GridInfo
{
    std::array<double, DIM> starts;  /// first grid line must outside of boundbox
    std::array<double, DIM> spaces;  /// grid space in each dim
    std::array<size_t, DIM> nsteps;  /// npixels (cells) in each axis
    std::array<double, DIM> mins;    /// boundbox min, may be extened if USE_CUBE_BOX is true
    std::array<double, DIM> maxs;
};

struct MeshData
{
    Handle(Poly_Triangulation) triangles;
    GridInfo grid_info;
    gp_Trsf local_transform;
};

/// normalized to aspect ratios all one
GridInfo generate_grid(const BoundBoxType bbox)
{
    std::array<scalar, DIM> mins, maxs;
    mins[0] = bbox[0], mins[1] = bbox[1], mins[2] = bbox[2];
    maxs[0] = bbox[3], maxs[1] = bbox[4], maxs[2] = bbox[5];
    std::vector<scalar> lengths = {maxs[0]-mins[0], maxs[1]-mins[1], maxs[2]-mins[2]};
    int m = 1;  //  margin extention on each side, must be integer 1 cell width, other value not tested
    std::array<double, DIM> spaces;

    if( USE_CUBE_BOX )
    {
        size_t imax = std::distance(lengths.cbegin(), std::max_element(lengths.cbegin(), lengths.cend()));
        auto maxL = lengths[imax];
        for(size_t i=0; i<DIM; i++)
        {
            if(imax != i)
            {
                scalar halfD = (lengths[imax] - lengths[i]) * 0.5;
                mins[i] -= halfD;
                maxs[i] += halfD;
            }
        }
        spaces = {maxL/(NGRIDS[0]-2*m), maxL/(NGRIDS[1]-2*m), maxL/(NGRIDS[2]-2*m)};
    }
    else
    {
        spaces = {lengths[0]/(NGRIDS[0]-2*m), lengths[1]/(NGRIDS[1]-2*m), lengths[2]/(NGRIDS[2]-2*m)};
    }

    std::array<double, DIM> starts = {mins[0] - spaces[0] * 0.5 * m, 
            mins[1] - spaces[1] * 0.5 * m, mins[2] - spaces[2] * 0.5 * m};  // first cell's center

    GridInfo gInfo{starts, spaces, {NGRIDS[0]-1, NGRIDS[1]-1, NGRIDS[2]-1},
        mins, maxs};  // 

    return gInfo;
}

BoundBoxType toBoundBox(const Bnd_Box& bbox)
{
    double xmin, xmax, ymin, ymax, zmin, zmax;
    bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    return {xmin, ymin, zmin, xmax, ymax, zmax};
}

BoundBoxType calcBoundBox(const TopoDS_Shape& shape)
{
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    return toBoundBox(bbox);
}

GridInfo generate_grid(const TopoDS_Shape& shape)
{
    // must be rotated with AABB = OBB,  AABB (Axis-align) is `class Bnd_Box`
    return generate_grid(calcBoundBox(shape));
}


TopoDS_Shape merge_geometry(const TopoDS_Shape& s)
{
    auto tol = 1e-4; // set 0.0 to temporally disable fuzzy operation
    TopTools_ListOfShape arguments;
    TopTools_ListOfShape tools;
    int count = 0;
    for (TopExp_Explorer anExp(s, TopAbs_SOLID); anExp.More(); anExp.Next())
    {
        if (anExp.Current().IsNull())
            throw OSD_Exception("one shape is null");
        
        if (count == 0)
        {
            if (tol > 0.0)
                // workaround for http://dev.opencascade.org/index.php?q=node/1056#comment-520
                arguments.Append(BRepBuilderAPI_Copy(anExp.Current()).Shape());
            else
                arguments.Append(anExp.Current());
        }
        else{
            if (tol > 0.0)
                // workaround for http://dev.opencascade.org/index.php?q=node/1056#comment-520
                tools.Append(BRepBuilderAPI_Copy(anExp.Current()).Shape());
            else
                tools.Append(anExp.Current()); 
        }
        
        count++;
    }
    if (count <= 1)
        return s;

    auto mkGFA = std::make_shared<BRepAlgoAPI_Fuse>();
    mkGFA->SetRunParallel(false);
    mkGFA->SetArguments(arguments);
    mkGFA->SetTools(tools);
    if (tol > 0.0)
        mkGFA->SetFuzzyValue(tol);
    mkGFA->SetNonDestructive(Standard_True);
    mkGFA->SetUseOBB(true);
    mkGFA->Build();
    if (!mkGFA->IsDone())
        throw OSD_Exception("General Fusion failed");

    return mkGFA->Shape();
}


/// currently brep only, assuming single solid
TopoDS_Shape read_geometry(std::string filename)
{
    TopoDS_Shape shape;
    if (filename.find(".brep") != std::string::npos or filename.find(".brp") != std::string::npos)
    {
        BRep_Builder builder;
        BRepTools::Read(shape, filename.c_str(), builder);
    }
    else{
        STEPControl_Reader aReader;
        IFSelect_ReturnStatus stat = aReader.ReadFile(filename.c_str());

        if (stat != IFSelect_RetDone)
        {
            throw std::runtime_error("step file read error for " + filename);
        }

        aReader.TransferRoots();   // translate all roots, return the number of transferred
        shape = aReader.OneShape(); // a compound if there are more than one shapes
    }
    return merge_geometry(shape);

}


/// for STEP, BREP shape input files, first of all, imprint and merge
/// rotate shape with OBB to AABB, can also translate
TopoDS_Shape  prepare_shape(std::string input, Bnd_OBB& obb)
{
    // 1. rotate view according to boundbox
    TopoDS_Shape shape = read_geometry(input);

    BRepBndLib::AddOBB(shape, obb);
    if(!obb.IsAABox()  &&  USE_OBB)
    {
        gp_Ax3 obb_ax(obb.Center(), obb.ZDirection(), obb.XDirection());
        // occt 7.3 does not have Position() to return gp_Ax3, so create this ax3
        gp_Trsf trsf;
        //gp_Ax3 gc_ax3; // the reference coordinate system (OXYZ)
        //gc_ax3.SetLocation(obb.Center());
        trsf.SetTransformation(obb_ax, gp::XOY());                            
        auto t = BRepBuilderAPI_Transform(shape, trsf, true);

        shape = t.Shape();  // it works
    }
    return shape;
}

/// free (used once in topology), not internal (embedded inside a solid)
std::shared_ptr<std::vector<TopoDS_Face>> get_free_faces(const TopoDS_Shape &shape) 
{
    auto freeFaces = std::make_shared<std::vector<TopoDS_Face>>();

    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        const TopoDS_Face &face = TopoDS::Face(ex.Current());
        // check if it is an external face, skip interior face

        // Important note: For the surface map, face equivalence is defined
        // by TopoDS_Shape::IsSame(), which ignores the orientation.
        // TopoDS_Shape::IsFree()
        //if (face.Orientation() == TopAbs_EXTERNAL)    
            freeFaces->push_back(face);
    }
    return freeFaces;
}

/// gp_Trsf& local_transform is an output parameter
const Handle(Poly_Triangulation)  generate_mesh(TopoDS_Face f, gp_Trsf& local_transform, double linDefl= 0.25)
{
    const double theAngDeflection = 0.5; //default 0.5, degree or radian?
    // mesh each face, by given best tols
    BRepMesh_IncrementalMesh facets(f, linDefl, true, theAngDeflection);

    TopLoc_Location local_loc;
    // BRep_Tool::Triangulation is for face only, return null if no mesh
    const Handle(Poly_Triangulation) triangles  = BRep_Tool::Triangulation(f, local_loc);
    // set output parameter: local transform
    local_transform = local_loc;
    // why TopoLoc_Location  can be assigned to gp_Trsf?

    if (triangles.IsNull()) {
        std::cout << "No facets for surface" << std::endl;
    }

    return triangles;
}

/// OFF STL surface mesh, OCCT 7.4
Handle(Poly_Triangulation) read_mesh(std::string filename)
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


bool write_mesh(const Handle(Poly_Triangulation)& t, std::string filename)
{
    if(t.IsNull())
        throw std::runtime_error(std::string("Mesh triangulation is empty, triangles")); 
    #if OCC_VERSION_HEX >= 0x070300
    return RWStl::WriteAscii(t, OSD_Path(filename.c_str()));   
    #else
    throw std::runtime_error(std::string("occ 7.4 is needed to read stl mesh as Poly_Triangulation"));
    #endif
}

BoundBoxType calc_mesh_boundbox(const Handle(Poly_Triangulation)& mesh)
{
    const TColgp_Array1OfPnt& nodes = mesh->Nodes();

    double xmin, xmax, ymin, ymax, zmin, zmax;

    // or using the first node coord to init bbox
    nodes(1).Coord(xmin, ymin, zmin);
    nodes(1).Coord(xmax, ymax, zmax);
    BoundBoxType bbox = {xmin, ymin, zmin, xmax, ymax, zmax};

    for (int i = nodes.Lower(); i <= nodes.Upper(); i++) {
        Standard_Real co[DIM];
        nodes(i).Coord(co[0], co[1], co[2]);
        for (int d = 0; d < DIM; d++)
        {
            if (bbox[d] > co[d])
                bbox[d] = co[d];
            if (bbox[d+DIM] < co[d])
                bbox[d+DIM] = co[d];          
        }
    }

   return bbox; 
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
    return n.Magnitude() /2.0;
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
    scalar     r, a, b;              // params to calc ray-plane intersect
    const scalar SMALL_NUM = 1e-12;    //

    // get triangle edge vectors and plane normal
    u = (*tri[1]) - (*tri[0]);
    v = (*tri[2]) - (*tri[0]);
    n = u.Crossed(v);                 // cross product
    if (n.Magnitude() < 1e-308)    // triangle is degenerate, zero area?
        return -1;                     // do not deal with this case
         
    w0 = *(R[0]) - (*tri[0]);
    gp_Vec dir = *(R[1]) - *(R[0]);
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

    scalar  uu, uv, vv, wu, wv, D;
    uu = u.Dot(u);
    uv = u.Dot(v);
    vv = v.Dot(v);
    w = I - (*tri[0]);
    wu = w.Dot(u);
    wv = w.Dot(v);
    D = uv * uv - uu * vv;   // D is a big negative number, if ray and triangle are almost parallel

    scalar precision = 1e-9;  // slightly outside the triangle may also counted as inside/on the edge of the triangle

    scalar s, t;
    s = (uv * wv - vv * wu) / D;   // s is very small/zero, if the ray is near a triangle vertex
    if (s < 0.0-precision || s > 1.0+precision)         // I is outside T
        return 0;
    t = (uv * wu - uu * wv) / D;   // if on edge/vertex, this t value is very small, zero
    if (t < 0.0-precision || (s + t) > 1.0+precision)  // I is outside T
        return 0;

    return 1;
}

// test if intersection in triangle edge situation, test precision sensitivity
void test_calc_intersection(bool near_vertex = false)
{
    //    Return: -1 = triangle is degenerate (a segment or point)
    //             0 =  disjoint (no intersect)
    //             1 =  intersect in unique point I1
    //             2 =  are in the same plane
    gp_Vec intersection_point;
    double tol = 1e-7;
    double BIG_NUM = 1e5;
    double heights[] = {BIG_NUM, 1, 1.0/BIG_NUM, -1.0/BIG_NUM -BIG_NUM};  
    // if height is a big number, the ray is almost parallel to the ray, if small then ray is normal to the triangle
    for(auto h: heights)
    {
        Coordinate p1(1, 0, 0);
        Coordinate p2(0, 1, 0);
        Coordinate p3(0, 0, h);
        Coordinate p12 = (p1 + p2) / 2.0;
        if(near_vertex)
        {
            p12 = p1;
        }

        Coordinate ray_start = p12;
        ray_start.SetZ(p12.Z() - 1);
        Coordinate ray_end = p12;
        ray_end.SetZ(p12.Z() + 1);
        const std::vector<const Coordinate*> tri = {&p1, &p2, &p3};
        const std::vector<const Coordinate*> ray = {&ray_start, &ray_end};


        double offsets[] = {-tol, 0, tol, tol*100};
        int results[] = {1, 1, 0, 0};
        double x_edge = p12.X();
        
        for(int i = 0; i<std::size(offsets); i++)
        {
            ray_start.SetX(x_edge + offsets[i]);
            ray_end.SetX(x_edge + offsets[i]);
            int ret = calc_intersection(tri, ray, intersection_point);
            if (ret != results[i])
            {
                std::cout << "Error: intersection type" << ret << " is not correct, should be " << results[i]
                    << ", height = " << h  << ", offset = " << offsets[i] << std::endl;
                std::cout << "intersection point coordinate: " << intersection_point.X() << ", "
                    << intersection_point.Y() << ", " << intersection_point.Z() << std::endl;
                calc_intersection(tri, ray, intersection_point);  // redo it, to debug
            }
        }
    }

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


void init_intersection_data(IntersectionData& data)
{

    data[0] = std::make_shared<IntersectionMat>(NGRIDS[0], NGRIDS[1]);
    data[1] = std::make_shared<IntersectionMat>(NGRIDS[1], NGRIDS[2]);
    data[2] = std::make_shared<IntersectionMat>(NGRIDS[2], NGRIDS[0]);

    for(int i=0; i<NVIEWS; i++)
    {
        IntersectionMat& mat = (*data[i]);
        init_matrix(mat);
    }
}

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

void normalize_intersection(IntersectionMat& mat, const std::pair<scalar, scalar> minmax)
{
    auto fh = minmax.second - minmax.first;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            auto& p = mat(r,c);
            if (p and p->size())
            {
                for(size_t i=0; i<p->size(); i++)
                    (*p)[i] = ((*p)[i] - minmax.first) / fh;
            }         
        }
    }
}


std::vector<std::array<size_t,2>> get_index_ranges(const std::vector<const Coordinate*>& polygon, const GridInfo gInfo)
{
    std::vector<std::array<size_t,2>> ranges;
    for(int iaxis = 0; iaxis < DIM; iaxis++)
    {
        std::vector<double> v;
        for(const auto& p: polygon)
        {
            v.push_back((*p).Coord(iaxis+1));   /// NOTE: index for X is 1 not 0
        }
        auto min_i = *std::min_element(v.cbegin(), v.cend());
        // what happen if the mesh vertex coincides with projection mesh, minus 1 to be conservative
        int imin = int((min_i - gInfo.starts[iaxis]) / gInfo.spaces[iaxis] ) - 1;
        if(imin < 0)  // comparison with zero will be always true,   size_t i= 0,  i-1  is a very big number
        {
            imin = 0;
        }
        auto max_i = *std::max_element(v.cbegin(), v.cend());
        /// should be rounded up for max, not to miss some point
        size_t imax = size_t((max_i - gInfo.starts[iaxis]) / gInfo.spaces[iaxis] ) + 1;
        if (imax > gInfo.nsteps[iaxis])
        {
            std::cout << "Error:  i_upper " << imax << " >= grid number " << gInfo.nsteps[iaxis] << std::endl;
            imax = gInfo.nsteps[iaxis];
        }
        ranges.push_back({static_cast<size_t>(imin), imax});
    }
    return ranges;
}

//const std::vector<Coordinate>& points, const std::array<int, 3>& tri
void insert_intersections(const std::vector<const Coordinate*>& triangle, const GridInfo& gInfo, IntersectionData& data)
{
    // get box bound for the tri, project to one plane, get possible indices from the grid
    const auto iranges = get_index_ranges(triangle, gInfo);  /// BUG solved, negative int becomes very big size_t number
    //std::vector<std::array<size_t,2>> iranges = {{0, NGRIDS[0]}, {0, NGRIDS[1]}, {0, NGRIDS[2]}};

    // if line and triangle intersection, then save the intersection coord's third-component
    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        const size_t r_start = iranges[r_index][0];
        const size_t r_end = iranges[r_index][1];
        size_t c_index = (1 + iaxis)==3? 0: 1 + iaxis;
        const size_t c_start = iranges[c_index][0];
        const size_t c_end = iranges[c_index][1];
        size_t third_index = (iaxis + 2)%3;
        const size_t t_start = iranges[third_index][0];
        const size_t t_end = iranges[third_index][1];
        gp_Vec dir(0, 0, 0);
        dir.SetCoord(third_index + 1, 1.0);
        // this ray must be bigger than boundbox range (min, max), it is fine here
        double third_min = gInfo.starts[third_index] - gInfo.spaces[third_index] * gInfo.nsteps[third_index];
        double third_max = gInfo.starts[third_index] + gInfo.spaces[third_index] * (gInfo.nsteps[third_index]*2);

        double h_min = gInfo.starts[third_index] - gInfo.spaces[third_index] * t_start;
        double h_max = gInfo.starts[third_index] + gInfo.spaces[third_index] * t_end;

        IntersectionMat& mat = *data[iaxis];
        for(size_t r = r_start; r <= r_end; r++)  // r_end must also been used, closed range
        {
            for(size_t c = c_start; c <= c_end; c++)
        // for(size_t r = 0; r <= NGRIDS[r_index]; r++)  // r_end must also been used, closed range
        // {
        //     for(size_t c = 0; c <= NGRIDS[c_index]; c++)
            {
                double r_value = gInfo.starts[r_index] + gInfo.spaces[r_index] * r;
                double c_value = gInfo.starts[c_index] + gInfo.spaces[c_index] * c;

                gp_Vec pos(0, 0, 0);  // occt index starts at 1, not zero
                pos.SetCoord(r_index+1, r_value);
                pos.SetCoord(c_index+1, c_value);
                pos.SetCoord(third_index + 1, third_min);
                gp_Vec p2 = pos;
                p2.SetCoord(third_index + 1, third_max);   // evil of copy-and-paste!
                //gp_Lin l(pos, dir);

                #if 0
                auto ret = has_intersection(triangle, r_value, c_value, iaxis);
                if(ret)
                {
                    gp_Vec I(0, 0, 0);
                    int m = calc_intersection(triangle,  {&pos, &p2}, I);
                    //if (m == 1)
                    mat(r, c)->push_back(I.Coord(third_index + 1));
                }
                #else
                gp_Vec I(0, 0, 0);
                int m = calc_intersection(triangle, {&pos, &p2}, I);
                if (m == 1)
                {
                    auto  h = I.Coord(third_index + 1);
                    //if (h < third_max && h > third_min)  // this may needs to be narrowed down.
                    if (h < h_max && h > h_min)
                    {
                        // if (r < r_start)
                        //     std::cout << " intersection found at r index = " << r << " less than r_start =" << r_start << std::endl;
                        // if (c < c_start)
                        //     std::cout << " intersection found at c index = " << c << " less than c_start = " << c_start << std::endl;
                        mat(r, c)->push_back(h);
                    }
                    else
                    {
                        std::cout << " Error: intersection point out of triangle bound box" << std::endl;
                    }
                    
                }
                #endif
            }
        }
    }
}

// tested by write the transformed trianglulation to stl file
Handle(Poly_Triangulation) transform_triangulation(const Handle(Poly_Triangulation) mesh, const gp_Trsf local_transform)
{
    const TColgp_Array1OfPnt& nodes = mesh->Nodes();

    TColgp_Array1OfPnt newPoints;
    newPoints.Resize(nodes.Lower(), nodes.Upper(), false);

    for (int i = nodes.Lower(); i <= nodes.Upper(); i++) {
        Standard_Real x, y, z;
        nodes(i).Coord(x, y, z);
        local_transform.Transforms(x, y, z);  // inplace modify
        newPoints(i) = gp_Pnt(x, y, z);
    }
   return new Poly_Triangulation(newPoints, mesh->Triangles());
}

void calc_intersections(const Handle(Poly_Triangulation) triangles, const GridInfo& ginfo, const gp_Trsf* local_transform, IntersectionData& data) 
{
    // mixing of array index starts at 1 and index starts 0 is buggy
    const TColgp_Array1OfPnt &nodes = triangles->Nodes();
    std::vector<Coordinate> points;   // NCollection_Array1<>
    points.reserve(nodes.Upper());
    if (nodes.Lower() == 1 )  // index starts at 1
    {
        //points.push_back({INFINITY, INFINITY, INFINITY});  // if we use index 0, then all inf
    }
    // retrieve triangle coordinates
    for (int i = nodes.Lower(); i <= nodes.Upper(); i++) {
        Standard_Real x, y, z;
        nodes(i).Coord(x, y, z);
        if(local_transform)   // this change coord, even suppose not to change
            local_transform->Transforms(x, y, z);  // inplace modify, 
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
        std::vector<const Coordinate*> triangle = {points.data() + indices[0]-1,  
                points.data() + indices[1]-1,  points.data() + indices[2]-1};
        insert_intersections(triangle, ginfo, data);
    }
}

