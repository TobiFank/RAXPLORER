import os

def list_files(start_path='.', exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'.next', 'node_modules', 'public'}

    for root, dirs, files in os.walk(start_path):
        # Remove directories that are in the exclude list
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')

        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')

# Replace '.' with your project's frontend path if needed
list_files('.')
