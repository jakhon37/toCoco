#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <nlohmann/json.hpp>
#include <filesystem>

using json = nlohmann::json; // for JSON
namespace fs = std::filesystem; // for filesystem operations

using namespace std;
using namespace cv;

class ToCoco {
public:
    ToCoco() {
        string root_path = ".";
        data_dir = root_path + "/test/filtered_data";  // for development only
        out_json = root_path + "/test/output.json";    // for development only
        categ = {"0"}; // Initialize with a single category "0"
        initCocoFormat();
    }

    void initCocoFormat() {
        string date_string = currentDate();
        coco_format = {
            {"info", {
                {"description", "Converted to COCO format for training with YOLOX model"},
                {"url", "no Dataset URL"},
                {"version", "1.0"},
                {"year", date_string.substr(0, 4)},
                {"contributor", "Mushtariy"},
                {"date_created", date_string}
            }},
            {"licenses", {{
                {"url", "no License URL"},
                {"id", 1},
                {"name", "no License Name"}
            }}},
            {"images", {}},
            {"annotations", {}},
            {"categories", {}}
        };

        // Add a default category
        for (int idx = 0; idx < categ.size(); ++idx) {
            coco_format["categories"].push_back({
                {"supercategory", "none"},
                {"id", idx + 1},
                {"name", categ[idx]}
            });
        }
    }

    vector<float> convertToCocoBbox(float x1, float y1, float x2, float y2, int img_width, int img_height) {
        float x1_abs = x1 * img_width;
        float y1_abs = y1 * img_height;
        float x2_abs = x2 * img_width;
        float y2_abs = y2 * img_height;
        float width = x2_abs - x1_abs;
        float height = y2_abs - y1_abs;
        return {x1_abs, y1_abs, width, height};
    }

    void cocoImage(string file_name, pair<int, int> img_size, int image_id) {
        coco_format["images"].push_back({
            {"license", 1},
            {"file_name", file_name},
            {"height", img_size.first},
            {"width", img_size.second},
            {"id", image_id}
        });
    }

    void cocoAnnot(int ann_id, int img_id, int class_id, vector<float> bbox) {
        coco_format["annotations"].push_back({
            {"id", ann_id},
            {"image_id", img_id},
            {"category_id", class_id + 1},
            {"bbox", bbox},
            {"area", bbox[2] * bbox[3]},
            {"iscrowd", 0}
        });
    }

    pair<int, int> mainCoco(const string& data_dir = "", const string& out_json = "", vector<string> categ = {"0"}, int img_id = 1, int ann_id = 1) {
        string current_data_dir = data_dir.empty() ? this->data_dir : data_dir;
        string current_out_json = out_json.empty() ? this->out_json : out_json;
        vector<string> current_categ = categ.empty() ? this->categ : categ;
        initCocoFormat();
        
        // Add categories to the format
        for (int idx = 0; idx < current_categ.size(); ++idx) {
            coco_format["categories"].push_back({
                {"supercategory", "none"},
                {"id", idx + 1},
                {"name", current_categ[idx]}
            });
        }

        for (const auto& entry : fs::directory_iterator(current_data_dir)) {
            string file = entry.path().string();
            if (file.find(".jpg") != string::npos) {
                Mat img = imread(file);
                int img_height = img.rows;
                int img_width = img.cols;

                string txt_path = file.substr(0, file.find_last_of(".")) + ".txt";

                if (fs::exists(txt_path)) {
                    ifstream txt_file(txt_path);
                    string line;
                    
                    cocoImage(file, {img_height, img_width}, img_id);
                    
                    while (getline(txt_file, line)) {
                        stringstream ss(line);
                        int class_id;
                        ss >> class_id;
                        float x1, y1, x2, y2;
                        ss >> x1 >> y1 >> x2 >> y2;

                        vector<float> bbox = convertToCocoBbox(x1, y1, x2, y2, img_width, img_height);
                        cocoAnnot(ann_id, img_id, class_id, bbox);
                        ann_id++;
                    }
                    img_id++;
                }
            }
        }
        
        ofstream outfile(current_out_json);
        outfile << json(coco_format).dump(4);
        return {img_id, ann_id};
    }

    void processDirectory(const string& data_dir = "") {
        string current_data_dir = data_dir.empty() ? this->data_dir : data_dir;
        fs::path out_dir = fs::path(current_data_dir) / "annotations";
        fs::create_directories(out_dir);
        int img_id = 1, ann_id = 1;
        int img_id_old = 1, ann_id_old = 1;

        vector<string> current_categ = categ.empty() ? vector<string>{"0"} : categ;  // Corrected to use vector of strings

        for (const auto& entry : fs::directory_iterator(current_data_dir)) {
            string item = entry.path().filename().string();
            string current_path = entry.path().string();

            if (fs::is_directory(current_path) && item.find("anno") == string::npos) {
                cout << "Number of files to process in " << item << ": " << (distance(fs::directory_iterator(current_path), fs::directory_iterator{}) / 2) << endl;  // Corrected to use distance
                fs::path out_json = out_dir / (item + ".json");

                tie(img_id, ann_id) = mainCoco(current_path, out_json.string(), current_categ, img_id, ann_id);
                initCocoFormat();
                cout << "Processed data <" << item << "> with image range: " << img_id_old << "-" << img_id << " and annotation range: " << ann_id_old << "-" << ann_id << endl;
                img_id_old = img_id;
                ann_id_old = ann_id;
            }
        }
    }



private:
    string data_dir;
    string out_json;
    vector<string> categ;
    json coco_format;

    string currentDate() {
        time_t now = time(0);
        tm* date = localtime(&now);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y/%m/%d", date);
        return string(buffer);
    }
};

int main() {
    ToCoco tcoco;
    string root_dir = "."; // Change to your desired root directory
    
    string data_dir = root_dir + "/data/coco";
    tcoco.processDirectory(data_dir);

    cout << "COCO conversion is done!" << endl;
    return 0;
}
