import torch
import argparse
import os

def inspect_pth_file(pth_path):
    if not os.path.isfile(pth_path):
        print(f"Το αρχείο {pth_path} δεν βρέθηκε.")
        return

    # Φορτώστε το .pth αρχείο
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Αποτυχία φόρτωσης του αρχείου {pth_path}: {e}")
        return

    # Ελέγξτε αν το checkpoint είναι ένα state_dict ή περιέχει περισσότερα στοιχεία
    if isinstance(checkpoint, dict):
        print(f"\nΤο αρχείο {pth_path} περιέχει τα παρακάτω κλειδιά:")
        for key in checkpoint.keys():
            print(f" - {key}")

        # Αν περιέχει state_dict, εκτυπώστε τα επίπεδα και τις διαστάσεις τους
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\nΠεριεχόμενο του state_dict:")
            for key, value in state_dict.items():
                print(f"Layer: {key} | Shape: {value.shape} | Type: {value.dtype}")
        else:
            # Αν δεν υπάρχει state_dict, υποθέτουμε ότι το checkpoint είναι ένα state_dict
            print("\nΠεριεχόμενο του checkpoint (state_dict):")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"Layer: {key} | Shape: {value.shape} | Type: {value.dtype}")
                else:
                    print(f"Key: {key} | Type: {type(value)}")
    else:
        print(f"\nΤο αρχείο {pth_path} δεν περιέχει ένα dict. Τύπος δεδομένων: {type(checkpoint)}")

def main():
    parser = argparse.ArgumentParser(description="Script για την εξέταση της δομής των .pth αρχείων.")
    parser.add_argument('pth_files', metavar='PATH', type=str, nargs='+',
                        help='Μονοπάτια προς τα .pth αρχεία που θέλετε να εξετάσετε')
    args = parser.parse_args()

    for pth_file in args.pth_files:
        print(f"\n--- Εξέταση αρχείου: {pth_file} ---")
        inspect_pth_file(pth_file)

if __name__ == "__main__":
    main()
