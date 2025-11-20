class BankAccount:
    def __init__(self):
        self.users = {
            '9800': {"Name": "Aloen", "Balance": 1000},
            '9801': {"Name": "Viran", "Balance": 1500},
            '9802': {"Name": "Neham", "Balance": 1000},
            '9803': {"Name": "Waqa", "Balance": 1500},
            '9804': {"Name": "Tutekin", "Balance": 1500},
            '9805': {"Name": "Saljan", "Balance": 1500}
        }

    def deposit(self):
        account_no = input("Enter your account number: ")
        if account_no not in self.users:
            print("Account not found.")
            return
        try:
            amount = float(input("Enter deposit amount: "))
            if amount <= 0:
                print("Amount must be positive.")
                return
            self.users[account_no]["Balance"] += amount
            print(f"Deposited {amount}. New balance: {self.users[account_no]['Balance']}")
        except ValueError:
            print("Invalid amount.")

    def withdraw(self):
        account_no = input("Enter your account number: ")
        if account_no not in self.users:
            print("Account not found.")
            return
        try:
            amount = float(input("Enter withdrawal amount: "))
            if amount <= 0:
                print("Amount must be positive.")
                return
            if amount > self.users[account_no]["Balance"]:
                print("Insufficient balance.")
                return
            self.users[account_no]["Balance"] -= amount
            print(f"Withdrawn {amount}. Remaining balance: {self.users[account_no]['Balance']}")
        except ValueError:
            print("Invalid amount.")

    def transfer(self):
        from_account = input("Enter your account number: ")
        to_account = input("Enter recipient account number: ")

        if from_account not in self.users:
            print("Your account not found.")
            return
        if to_account not in self.users:
            print("Recipient account not found.")
            return
        try:
            amount = float(input("Enter transfer amount: "))
            if amount <= 0:
                print("Amount must be positive.")
                return
            if amount > self.users[from_account]["Balance"]:
                print("Insufficient balance.")
                return
            self.users[from_account]["Balance"] -= amount
            self.users[to_account]["Balance"] += amount
            print(f"Transferred {amount} to {to_account}. Remaining balance: {self.users[from_account]['Balance']}")
        except ValueError:
            print("Invalid amount.")

class Customer:
    def __init__(self, users):
        self.users = users

    def show_users(self):
        print("\n----- Current Users -----")
        for account_no, details in self.users.items():
            print(f"Account No: {account_no}, Name: {details['Name']}, Balance: {details['Balance']}")
        print("--------------------------\n")

    def open_account(self):
        new_id = input("Enter new account number: ")
        if new_id in self.users:
            print("Account number already exists.")
            return
        new_name = input("Enter account holder's name: ")
        try:
            initial_deposit = float(input("Enter initial deposit amount: "))
            if initial_deposit < 0:
                print("Deposit must be non-negative.")
                return
            self.users[new_id] = {"Name": new_name, "Balance": initial_deposit}
            print(f"Account {new_id} opened for {new_name} with balance {initial_deposit}")
        except ValueError:
            print("Invalid amount entered.")


    def close_account(self):
        del_id = input("Enter account number to close: ")
        if del_id in self.users:
            del self.users[del_id]
            print(f"Account {del_id} closed successfully.")
        else:
            print("Account not found.")

def main():
    bank = BankAccount()
    customer = Customer(bank.users)

    while True:
        print("\n===== Bank Menu =====")
        print("1. Deposit")
        print("2. Withdraw")
        print("3. Transfer")
        print("4. Show Users")
        print("5. Open Account")
        print("6. Close Account")
        print("7. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            bank.deposit()
        elif choice == "2":
            bank.withdraw()
        elif choice == "3":
            bank.transfer()
        elif choice == "4":
            customer.show_users()
        elif choice == "5":
            customer.open_account()
        elif choice == "6":
            customer.close_account()
        elif choice == "7":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()