# 读取 json 文件, 将其中 labels 字段的值转化为数值 label 并保存为新的 json 文件
# 下面是一个示例, 你需要将 "labels" 中的 list 转化为数值 label list, "Yes" 对应 1, "No" 对应 -1, 其他标签对应 0
# {
#     "problem": "Given that the work function of chromium is $4.40 \\mathrm{eV}$, calculate the kinetic energy of electrons in Joules emitted from a clean chromium surface that is irradiated with ultraviolet radiation of wavelength $200 \\mathrm{~nm}$.",
#     "responses": [
#       "To determine the kinetic energy of electrons emitted from a clean chromium surface when irrational with ultraviolet radiation of wavelength \\(200 \\mathrm{~nm}\\), we can use the photoelectric effect equation. The photoelectric effect equation is given by:\n\n\\[ E_k = \\frac{hc}{\\lambda} - \\phi \\]\n\nwhere:\n- \\( E_k \\) is the kinetic energy of the emitted electrons,\n- \\( h \\) is Planck's constant (\\(6.626 \\times 10^{-34} \\mathrm{~J \\cdot s}\\)),\n- \\( c \\) is the speed of light (\\(3.00 \\times 10^8 \\mathrm{~m/s}\\)),\n- \\( \\lambda \\) is the wavelength of the incident light,\n- \\( \\phi \\) is the work function of the material.\n\nFirst, we need to convert the wavelength from nanometers to meters:\n\n\\[ \\lambda = 200 \\mathrm{~nm} = 200 \\times 10^{-9} \\mathrm{~m} = 2 \\times 10^{-7} \\mathrm{~m} \\]\n\nNext, we substitute the values of \\( h \\), \\( c \\), \\( \\lambda \\), and \\( \\phi \\) into the photoelectric effect equation. The work function of chromium is given as \\( 4.40 \\mathrm{~eV} \\). We need to convert this energy from electron-volts to Joules. Since \\( 1 \\mathrm{~eV} = 1.602 \\times 10^{-19} \\mathrm{~J} \\), we have:\n\n\\[ \\phi = 4.40 \\mathrm{~eV} = 4.40 \\times 1.602 \\times 10^{-19} \\mathrm{~J} = 7.05 \\times 10^{-19} \\mathrm{~J} \\]\n\nNow, we substitute the values into the photoelectric effect equation:\n\n\\[ E_k = \\frac{(6.626 \\times 10^{-34} \\mathrm{~J \\cdot s})(3.00 \\times 10^8 \\mathrm{~m/s})}{2 \\times 10^{-7} \\mathrm{~m}} - 7.05 \\times 10^{-19} \\mathrm{~J} \\]\n\nFirst, we calculate the numerator:\n\n\\[ (6.626 \\times 10^{-34} \\mathrm{~J \\cdot s})(3.00 \\times 10^8 \\mathrm{~m/s}) = 1.9878 \\times 10^{-25} \\mathrm{~J \\cdot m} \\]\n\nNext, we divide by the wavelength:\n\n\\[ \\frac{1.9878 \\times 10^{-25} \\mathrm{~J \\cdot m}}{2 \\times 10^{-7} \\mathrm{~m}} = 9.939 \\times 10^{-19} \\mathrm{~J} \\]\n\nFinally, we subtract the work function:\n\n\\[ E_k = 9.939 \\times 10^{-19} \\mathrm{~J} - 7.05 \\times 10^{-19} \\mathrm{~J} = 2.889 \\times 10^{-19} \\mathrm{~J} \\]\n\nRounding to three significant figures, we get:\n\n\\[ E_k = 2.89 \\times 10^{-19} \\mathrm{~J} \\]\n\nTherefore, the kinetic energy of the electrons emitted is:\n\n\\[ \\boxed{2.89 \\times 10^{-19} \\mathrm{~J}} \\]"
#     ],
#     "answer": "2.88e-19",
#     "labels": [
#       "No"
#     ]
# }
import json
import os
import argparse
from tqdm import tqdm

def transform_format(input_file: str, output_file: str):
    """Transform the format of the JSON file by converting labels to numerical values."""
    label_mapping = {
        "Yes": 1,
        "No": -1
    }

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformed_data = []
    for item in tqdm(data, desc="Transforming data"):
        transformed_item = {
            "problem": item["problem"],
            "responses": item["responses"],
            "answer": item["answer"],
            "labels": [label_mapping.get(label, 0) for label in item["labels"]]  # Default to 0 for other labels
        }
        transformed_data.append(transformed_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

    print(f"Transformed data saved to {output_file}")
    

if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Transform JSON format of math problem responses")
    parser.add_argument("--filename", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    transform_format(args.filename, args.output)