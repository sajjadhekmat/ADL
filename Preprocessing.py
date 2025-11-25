import pandas as pd
import os
import glob

# 🔹 Base folders
base_input = '/Users/sajjadhekmat/PycharmProjects/ADL/Data'
base_output = '/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData'

# 🔹 Loop through subfolders (1 → 12)
for i in range(1, 13):
    input_folder = os.path.join(base_input, str(i))
    output_folder = os.path.join(base_output, str(i))

    os.makedirs(output_folder, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    for file in csv_files:
        print(f"📂 Processing {file} ...")

        # --- Read the CSV file ---
        df = pd.read_csv(file, skiprows=2)

        # --- Fix 'Unnamed' columns ---
        new_columns = []
        current_name = None
        for col in df.columns:
            if "Unnamed" not in col:
                current_name = col
                new_columns.append(col)
            else:
                new_columns.append(current_name)
        df.columns = new_columns



        # --- Remove the part before colon (and the colon itself) in all column names ---
        df.columns = df.columns.str.replace(r'^.*?:', '', regex=True)


        # --- Merge column names with first row ---
        df.columns = [f"{col}_{val}" for col, val in zip(df.columns, df.iloc[0])]
        df.columns = df.columns.str.replace(' ', '_').str.lower()



        # --- Drop first two rows and last column ---
        df = df.drop(0).reset_index(drop=True)
        df = df.drop(0).reset_index(drop=True)
        df = df.drop(df.columns[-1], axis=1)

        # --- Keep only arm-related columns ---
        cols_to_keep = [
            'none_frame', 'none_sub_frame',
            'shoulderr_x', 'shoulderr_y', 'shoulderr_z',
            *[f'humr{i}_{axis}' for i in range(1, 9) for axis in ['x', 'y', 'z']],
            *[f'forearmr{i}_{axis}' for i in range(1, 9) for axis in ['x', 'y', 'z']],
            *[f'handr{i}_{axis}' for i in range(1, 6) for axis in ['x', 'y', 'z']],
            'c7_x', 'c7_y', 'c7_z',
            'strtop_x', 'strtop_y', 'strtop_z',
            'strbottom_x', 'strbottom_y', 'strbottom_z',
            't8_x', 't8_y', 't8_z',
            'torsoside_x', 'torsoside_y', 'torsoside_z',
            'pelvisbl_x', 'pelvisbl_y', 'pelvisbl_z',
            'pelvisbr_x', 'pelvisbr_y', 'pelvisbr_z',
            'pelvisfl_x', 'pelvisfl_y', 'pelvisfl_z',
            'pelvisfr_x', 'pelvisfr_y', 'pelvisfr_z'
        ]

        df = df[[c for c in cols_to_keep if c in df.columns]]

        # --- ✅ Rename first two columns ---
        df = df.rename(columns={
            'none_frame': 'frame',
            'none_sub_frame': 'sub_frame'
        })

        # --- Save cleaned file ---
        file_name = os.path.basename(file)
        save_path = os.path.join(output_folder, f"cleaned_{file_name}")
        df.to_csv(save_path, index=False)

        print(f"✅ Saved cleaned file to: {save_path}")

print("🎉 All subfolders processed successfully!")
