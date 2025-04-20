import os
import shutil

def merge_datasets(dataset1, dataset2, result_dataset):
    for subset in ['train', 'valid']:
        subset_path1 = os.path.join(dataset1, subset)
        subset_path2 = os.path.join(dataset2, subset)
        subset_result = os.path.join(result_dataset, subset)
        os.makedirs(subset_result, exist_ok=True)

        for base_path in [subset_path1, subset_path2]:
            if not os.path.exists(base_path):
                continue  # Si una de las carpetas no existe, la saltamos

            for class_name in os.listdir(base_path):
                class_src = os.path.join(base_path, class_name)
                class_dst = os.path.join(subset_result, class_name)

                if not os.path.isdir(class_src):
                    continue

                os.makedirs(class_dst, exist_ok=True)

                for file_name in os.listdir(class_src):
                    src_file = os.path.join(class_src, file_name)
                    dst_file = os.path.join(class_dst, file_name)

                    # Evitar sobrescribir imágenes con mismo nombre
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(file_name)
                        i = 1
                        while os.path.exists(dst_file):
                            dst_file = os.path.join(class_dst, f"{base}_{i}{ext}")
                            i += 1

                    shutil.copy2(src_file, dst_file)
                    print(f"Copiado: {src_file} → {dst_file}")


