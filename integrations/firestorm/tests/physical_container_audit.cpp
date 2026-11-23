// Native Firestorm import, physical-container and asset round-trip regression.
#include "lldaeloader.h"
#include "llapr.h"
#include "llcontrol.h"
#include "llconvexdecomposition.h"
#include "llerrorcontrol.h"
#include "llsdserialize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

LLControlGroup gSavedSettings("saved");

class LLModelPreview
{
public:
    static constexpr const char* sSuffixVarNames[5] = {
        "FSMeshLowestLodSuffix", "FSMeshLowLodSuffix", "FSMeshMediumLodSuffix",
        "FSMeshHighLodSuffix", "FSMeshPhysicsSuffix"};
};

std::string stripSuffix(std::string name)
{
    // <FS:Beq> Selectable suffixes
    //if ((name.find("_LOD") != -1) || (name.find("_PHYS") != -1))
    // {
    //     return name.substr(0, name.rfind('_'));
    // }
    for(int i=0; i < LLModel::NUM_LODS; i++)
    {
        const auto& suffix = gSavedSettings.getString(LLModelPreview::sSuffixVarNames[i]);
        if (suffix.size() && name.find(suffix) != std::string::npos)
        {
            return name.substr(0, name.rfind('_'));
        }
    } // </FS:Beq>
    return name;
}



static const LODSuffixArray kSuffixes{"LOD0", "LOD1", "LOD2", "", "PHYS"};

static void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

static LLModelInstance& singleInstance(LLModelLoader::scene& scene, LLModel* model)
{
    LLModelInstance* found = nullptr;
    for (auto& entry : scene)
    {
        for (auto& instance : entry.second)
        {
            if (instance.mModel.get() == model)
            {
                require(!found, "Fixture model was instanced more than once");
                found = &instance;
            }
        }
    }
    require(found, "Fixture model has no scene owner");
    return *found;
}

static LLModel::hull copyHull(const LLCDHull& raw)
{
    require(raw.mVertexBase && raw.mVertexStrideBytes >= 3 * sizeof(float),
            "Backend returned invalid hull storage");
    auto* point = reinterpret_cast<const char*>(raw.mVertexBase);
    LLModel::hull output;
    for (int i = 0; i < raw.mNumVertices; ++i, point += raw.mVertexStrideBytes)
    {
        std::array<float, 3> xyz;
        std::memcpy(xyz.data(), point, sizeof(xyz));
        output.emplace_back(xyz.data());
    }
    return output;
}

#include "../lodestar_firestorm_decomp.hpp"
#include <filesystem>

struct Corner { LLVector3 point, normal; LLVector2 uv; };
using Triangle = std::array<Corner,3>;
static std::vector<Triangle> triangles(LLModel* model, const LLMatrix4& transform) {
    LLVector3 basis[3] = {LLVector3(transform.mMatrix[0]),LLVector3(transform.mMatrix[1]),LLVector3(transform.mMatrix[2])};
    F32 scale[3]; for(int axis=0;axis<3;++axis) scale[axis]=basis[axis].normalize();
    std::vector<Triangle> result;
    for(int f=0;f<model->getNumVolumeFaces();++f) {
        const auto& face=model->getVolumeFace(f);
        require(face.mNumIndices%3==0,"Invalid visual index stream");
        for(int i=0;i<face.mNumIndices;i+=3) {
            Triangle triangle;
            for(int c=0;c<3;++c) {
                const int v=face.mIndices[i+c]; require(v<face.mNumVertices,"Missing visual vertex");
                triangle[c].point=LLVector3(face.mPositions[v].getF32ptr())*transform;
                require(face.mNormals && face.mTexCoords,"Visual attributes lost");
                triangle[c].normal.clear();
                for(int a=0;a<3;++a) triangle[c].normal+=basis[a]*(face.mNormals[v][a]/scale[a]);
                triangle[c].normal.normalize(); triangle[c].uv=face.mTexCoords[v];
                require(llfinite(triangle[c].uv[0]) && llfinite(triangle[c].uv[1]),"Nonfinite decoded UV");
            }
            require(((triangle[1].point-triangle[0].point)%(triangle[2].point-triangle[0].point)).magVecSquared()>0,"Degenerate visual triangle");
            result.push_back(triangle);
        }
    }
    return result;
}
static double match(const std::vector<Triangle>& expected,const std::vector<Triangle>& actual,double tolerance) {
    require(expected.size()==actual.size(),"Physical frame changed visual triangle count");
    std::vector<bool> used(expected.size()); double maximum=0;
    for(const auto& face:actual) {
        bool found=false;
        for(size_t i=0;i<expected.size() && !found;++i) if(!used[i]) {
            for(int shift=0;shift<3 && !found;++shift) {
                bool good=true; double error=0;
                for(int c=0;c<3;++c) {
                    const auto& reference=expected[i][(c+shift)%3];
                    const double distance=(reference.point-face[c].point).magVec(); error=std::max(error,distance);
                    good &= distance<=tolerance && (reference.uv-face[c].uv).length()<5e-5f &&
                        reference.normal*face[c].normal>std::cos(.1*3.141592653589793/180.);
                }
                if(good) { found=true; used[i]=true; maximum=std::max(maximum,error); }
            }
        }
        require(found,"Packed surface, winding, UV or world normal differs from authored reference");
    }
    return maximum;
}
int main(int argc,char**argv) {
    try {
        require(argc>=3 && argc<=5,"Expected package folder, JSON report, optional import scale and normal setting");
        const F32 importScale=argc>=4 ? std::stof(argv[3]) : 1.f;
        const bool normalFixup=argc==5 && std::string(argv[4])=="true";
        LLMatrix4 importTransform; importTransform.initScale(LLVector3(importScale,importScale,importScale));
        ll_init_apr(); LLError::initForApplication(".",".");
        gSavedSettings.declareBOOL("FSMeshImportScaleFixup",false,"Native normal transform");
        for(size_t i=0;i<kSuffixes.size();++i) gSavedSettings.declareString(LLModelPreview::sSuffixVarNames[i],kSuffixes[i],"LOD suffix");
        JointTransformMap joints; JointNameSet names; std::map<std::string,std::string,std::less<>> aliases;
        auto state=[](U32,void*){};
        std::array<std::unique_ptr<LLDAELoader>,5> loaders;
        std::array<std::unique_ptr<LLDAELoader>,4> references;
        const std::array<std::string,5> suffix={"_LOD0","_LOD1","_LOD2","","_PHYS"};
        for(int lod=0;lod<5;++lod) {
            auto path=(std::filesystem::path(argv[1])/("Lodestar"+suffix[lod]+".dae")).string();
            gSavedSettings.setBOOL("FSMeshImportScaleFixup",normalFixup);
            loaders[lod]=std::make_unique<LLDAELoader>(path,lod,nullptr,nullptr,nullptr,state,nullptr,joints,names,aliases,110,192,0,true,kSuffixes);
            require(loaders[lod]->OpenFile(path) && !loaders[lod]->mModelList.empty(),"Cannot import physical-frame package");
            if(lod<4) {
                gSavedSettings.setBOOL("FSMeshImportScaleFixup",false);
                path=(std::filesystem::path(argv[1])/("Reference"+suffix[lod]+".dae")).string();
                references[lod]=std::make_unique<LLDAELoader>(path,lod,nullptr,nullptr,nullptr,state,nullptr,joints,names,aliases,110,192,0,true,kSuffixes);
                require(references[lod]->OpenFile(path),"Cannot import reference");
            }
        }
        require(LLConvexDecomposition::initSystem()==LLCD_OK && LLConvexDecomposition::initThread()==LLCD_OK,"Cannot initialize physics backend");
        auto* backend=LLConvexDecomposition::getInstance();
        std::ofstream output(argv[2]); output<<std::setprecision(12)<<"{\"passed\":true,\"models\":["; bool first=true;
        for(auto& high:loaders[3]->mModelList) {
            std::array<LLModel*,5> models{};
            for(int lod=0;lod<5;++lod) {
                const auto wanted=high->mLabel+(lod==3 ? "" : "_"+kSuffixes[lod]);
                auto found=std::find_if(loaders[lod]->mModelList.begin(),loaders[lod]->mModelList.end(),[&](const auto&m){return m->mLabel==wanted;});
                require(found!=loaders[lod]->mModelList.end(),"Missing matching tier"); models[lod]=found->get();
                require(models[lod]->mLodestarPhysicalContainer==high->mLodestarPhysicalContainer,"Frame differs across LODs");
            }
            auto owner=singleInstance(loaders[3]->mScene,high);
            owner.mTransform*=importTransform;
            if(high->mLodestarPhysicalContainer) for(int a=0;a<3;++a)
                require(LLVector3(owner.mTransform.mMatrix[a]).magVec()>=.01f,"Physical instance dimension below minimum");
            LLModel::convex_hull_decomposition hulls;
            auto* physics=models[4];
            if(high->mLodestarPhysicalContainer) {
                LLVector3 scale,center; physics->getNormalizedScaleTranslation(scale,center);
                std::vector<LLVector3> points; std::vector<U16> indices;
                for(int f=0;f<physics->getNumVolumeFaces();++f) {
                    const auto& face=physics->getVolumeFace(f); const auto offset=points.size();
                    for(int j=0;j<face.mNumVertices;++j) points.emplace_back(face.mPositions[j].getF32ptr());
                    for(int j=0;j<face.mNumIndices;++j) indices.push_back(U16(offset+face.mIndices[j]));
                }
                int id; backend->genDecomposition(id); backend->bindDecomposition(id);
                backend->setParam("Lodestar Metric X",scale[0]); backend->setParam("Lodestar Metric Y",scale[1]); backend->setParam("Lodestar Metric Z",scale[2]);
                LLCDMeshData data{points.front().mV,sizeof(LLVector3),int(points.size()),indices.data(),LLCDMeshData::INT_16,3*sizeof(U16),int(indices.size()/3)};
                require(backend->setMeshData(&data,false)==LLCD_OK && backend->executeStage(0)==LLCD_OK,"Physical container Analyze failed");
                require(backend->getNumHullsFromStage(0)==1,"Physical box did not produce one hull");
                LLCDHull hull{}; require(backend->getHullFromStage(0,0,&hull)==LLCD_OK,"Missing box hull");
                hulls.push_back(copyHull(hull)); require(hulls.front().size()==8,"Physical box is not eight vertices");
                physics->setConvexHullDecomposition(hulls,{}); physics->mPhysics.mBaseHull=hulls.front(); backend->deleteDecomposition(id);
            }
            std::stringstream packed;
            const auto header=LLModel::writeModel(packed,physics,models[3],models[2],models[1],models[0],physics->mPhysics,false,false,false,LLModel::WRITE_BINARY,false,0);
            if(high->mLodestarPhysicalContainer) require(header["physics_mesh"]["size"].asInteger()==0 && header["physics_convex"]["size"].asInteger()>0,"Box did not serialize as hull physics");
            if(!first) output<<','; first=false;
            output<<"{\"name\":"<<std::quoted(high->mLabel)<<",\"physical_container\":"<<(high->mLodestarPhysicalContainer?"true":"false")<<",\"lods\":[";
            for(int lod=0;lod<4;++lod) {
                auto ref=std::find_if(references[lod]->mModelList.begin(),references[lod]->mModelList.end(),[&](const auto&m){return m->mLabel==models[lod]->mLabel;});
                require(ref!=references[lod]->mModelList.end(),"Missing reference model");
                require(models[lod]->mMaterialList==(*ref)->mMaterialList,"Physical frame changed material slots");
                auto refOwner=singleInstance(references[lod]->mScene,ref->get());
                refOwner.mTransform*=importTransform;
                const auto expected=triangles(ref->get(),refOwner.mTransform);
                match(expected,triangles(models[lod],owner.mTransform),1e-7);
                LLVolumeParams params; params.setType(LL_PCODE_PROFILE_SQUARE,LL_PCODE_PATH_LINE);
                LLPointer<LLModel> decoded=new LLModel(params,float(lod)); packed.clear(); packed.seekg(0);
                require(decoded->loadModel(packed),"Packed LOD could not be decoded");
                const auto actual=triangles(decoded,owner.mTransform);
                const double maximum=match(expected,actual,2e-6);
                if(high->mLodestarPhysicalContainer && lod==3) {
                    require(decoded->mPhysics.mHull.size()==1 && decoded->mPhysics.mHull[0].size()==8,"Packed physical box lost vertices");
                    const auto& hull=decoded->mPhysics.mHull[0]; bool volume=false;
                    for(size_t a=1;a<hull.size();++a) for(size_t b=a+1;b<hull.size();++b) for(size_t c=b+1;c<hull.size();++c)
                        volume |= std::abs((hull[a]-hull[0])*((hull[b]-hull[0])%(hull[c]-hull[0])))>0;
                    require(volume,"Packed physical box has zero volume");
                }
                if(lod) output<<',';
                output<<"{\"lod\":"<<lod<<",\"triangles\":"<<actual.size()<<",\"maximum_world_displacement_m\":"<<maximum<<",\"degenerate_triangles\":0,\"uv_and_world_normals_preserved\":true}";
            }
            output<<"]}";
        }
        output<<"],\"gui_validated\":false,\"simulator_validated\":false}\n";
        require(output.good(),"Cannot write validation report");
        LLConvexDecomposition::quitThread(); LLConvexDecomposition::quitSystem();
        std::cout<<"Physical frames, all visual LODs and native packed collider passed\n";
        return 0;
    } catch(const std::exception&error) {std::cerr<<error.what()<<'\n';return 1;}
}
