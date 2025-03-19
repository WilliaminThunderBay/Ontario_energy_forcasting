import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class EnergyPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ontario Energy Demand System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f5f5f5")
        
        # Track menu state
        self.menu_visible = False
        self.current_page = None
        
        # Top bar mimicking Ontario website style
        self.top_frame = tk.Frame(root, bg="#000000", height=100)
        self.top_frame.pack(fill="x", side="top")

        try:
            self.icon_image = Image.open("1.png").resize((160, 60))
            self.icon = ImageTk.PhotoImage(self.icon_image)
            self.icon_label = tk.Label(self.top_frame, image=self.icon, bg="#000000")
            self.icon_label.pack(side="left", padx=20)
        except Exception as e:
            messagebox.showwarning("Warning", "Icon image not found.")
        
        # Load both icons at initialization
        try:
            self.menu_icon_image = Image.open("2.png").resize((160, 60))
            self.menu_icon = ImageTk.PhotoImage(self.menu_icon_image)
            
            self.close_icon_image = Image.open("3.png").resize((160, 60))
            self.close_icon = ImageTk.PhotoImage(self.close_icon_image)
            
            # Start with menu icon
            self.right_icon_label = tk.Label(self.top_frame, image=self.menu_icon, bg="#000000", cursor="hand2")
            self.right_icon_label.pack(side="right", padx=20)
            
            # Store icon width for dropdown sizing
            self.icon_width = self.menu_icon_image.width
            
            # Add hover effect to change icon
            self.right_icon_label.bind("<Enter>", self.on_icon_enter)
            self.right_icon_label.bind("<Leave>", self.on_icon_leave)
            self.right_icon_label.bind("<Button-1>", self.toggle_dropdown_menu)
        except Exception as e:
            messagebox.showwarning("Warning", f"Icon images not found: {e}")
        
        # Green bar below the black top bar
        self.green_bar = tk.Frame(root, bg="#006A4D", height=40)
        self.green_bar.pack(fill="x", side="top")
        
        self.green_label = tk.Label(self.green_bar, text="Ontario Energy Forecasting", fg="white", bg="#006A4D", font=("Arial", 14, "bold"))
        self.green_label.pack(side="left", padx=20, pady=5)
        
        # Create dropdown menu with increased visibility and proper zindex
        self.dropdown_frame = tk.Frame(self.root, bg="white", relief="solid", borderwidth=2)
        self.dropdown_buttons = []
        
        options = ["Train", "Predict", "Evaluation"]
        for option in options:
            btn = tk.Button(self.dropdown_frame, text=option, anchor="w", relief="flat", 
                          bg="white", fg="black", width=20, height=2, font=("Arial", 10),
                          command=lambda opt=option: self.handle_menu_selection(opt))
            btn.pack(fill="x", padx=10, pady=5)
            self.dropdown_buttons.append(btn)
            
            # Add hover effect to menu items
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#f0f0f0"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="white"))
        
        # Add event to close menu when clicking elsewhere
        self.root.bind("<Button-1>", self.close_menu_if_outside)
        
        # Create frames for different sections
        self.create_section_frames()
        
        # Welcome message (initial display)
        self.welcome_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.welcome_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        welcome_label = tk.Label(self.welcome_frame, text="Welcome to Ontario Energy Demand System", 
                                font=("Arial", 18, "bold"), bg="#f5f5f5")
        welcome_label.pack(pady=10)
        
        instruction_label = tk.Label(self.welcome_frame, text="Please select an option from the menu to continue", 
                                    font=("Arial", 12), bg="#f5f5f5")
        instruction_label.pack(pady=5)
        
        # Show welcome screen initially
        self.show_page("Welcome")
    
    def create_section_frames(self):
        """Create frames for Train, Predict, and Evaluation sections"""
        # Content container
        self.content_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.content_frame.pack(fill="both", expand=True, pady=20)
        
        # Train section
        self.train_frame = tk.Frame(self.content_frame, bg="#f5f5f5")
        train_label = tk.Label(self.train_frame, text="Training Configuration", font=("Arial", 16, "bold"), bg="#f5f5f5")
        train_label.pack(pady=10)
        
        train_config_frame = ttk.Frame(self.train_frame, padding=20, relief="ridge")
        train_config_frame.pack(pady=10)
        
        ttk.Label(train_config_frame, text="Data Source:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        data_source_var = tk.StringVar()
        data_source_combo = ttk.Combobox(train_config_frame, textvariable=data_source_var, 
                                       values=["IESO Historical Data", "Weather Data", "Economic Indicators"], 
                                       state='readonly', width=30)
        data_source_combo.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(train_config_frame, text="Algorithm:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        algorithm_var = tk.StringVar()
        algorithm_combo = ttk.Combobox(train_config_frame, textvariable=algorithm_var, 
                                      values=["ARIMA", "XGBoost", "LSTM", "Ensemble"], 
                                      state='readonly', width=30)
        algorithm_combo.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(train_config_frame, text="Training Period:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        period_var = tk.StringVar()
        period_combo = ttk.Combobox(train_config_frame, textvariable=period_var, 
                                   values=["Last 1 Year", "Last 5 Years", "Last 10 Years"], 
                                   state='readonly', width=30)
        period_combo.grid(row=2, column=1, padx=10, pady=5)
        
        train_button = ttk.Button(train_config_frame, text="Start Training")
        train_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Predict section (the original prediction form)
        self.predict_frame = tk.Frame(self.content_frame, bg="#f5f5f5")
        predict_label = tk.Label(self.predict_frame, text="Prediction Configuration", font=("Arial", 16, "bold"), bg="#f5f5f5")
        predict_label.pack(pady=10)
        
        predict_config_frame = ttk.Frame(self.predict_frame, padding=20, relief="ridge")
        predict_config_frame.pack(pady=10)
        
        ttk.Label(predict_config_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(predict_config_frame, textvariable=self.model_var, 
                                        values=["ARIMA", "XGBoost", "LSTM", "ARIMA + LSTM + XGBoost"], 
                                        state='readonly', width=30)
        self.model_combo.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(predict_config_frame, text="Prediction Target:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(predict_config_frame, textvariable=self.target_var, 
                                         values=["Electricity Demand", "Price"], 
                                         state='readonly', width=30)
        self.target_combo.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(predict_config_frame, text="Forecast Duration:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.duration_var = tk.StringVar()
        self.duration_combo = ttk.Combobox(predict_config_frame, textvariable=self.duration_var, 
                                           values=["2 Years", "5 Years", "10 Years"], 
                                           state='readonly', width=30)
        self.duration_combo.grid(row=2, column=1, padx=10, pady=5)
        
        self.run_button = ttk.Button(predict_config_frame, text="Run Prediction", command=self.run_prediction)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Result output for prediction
        self.output_text = tk.Text(self.predict_frame, height=12, width=80, wrap="word", state="disabled")
        self.output_text.pack(pady=10)
        
        # Evaluation section
        self.evaluation_frame = tk.Frame(self.content_frame, bg="#f5f5f5")
        evaluation_label = tk.Label(self.evaluation_frame, text="Model Evaluation", font=("Arial", 16, "bold"), bg="#f5f5f5")
        evaluation_label.pack(pady=10)
        
        eval_config_frame = ttk.Frame(self.evaluation_frame, padding=20, relief="ridge")
        eval_config_frame.pack(pady=10)
        
        ttk.Label(eval_config_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        eval_model_var = tk.StringVar()
        eval_model_combo = ttk.Combobox(eval_config_frame, textvariable=eval_model_var, 
                                      values=["ARIMA", "XGBoost", "LSTM", "Ensemble"], 
                                      state='readonly', width=30)
        eval_model_combo.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(eval_config_frame, text="Evaluation Metric:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        metric_var = tk.StringVar()
        metric_combo = ttk.Combobox(eval_config_frame, textvariable=metric_var, 
                                   values=["RMSE", "MAE", "MAPE", "All Metrics"], 
                                   state='readonly', width=30)
        metric_combo.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(eval_config_frame, text="Test Period:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        test_period_var = tk.StringVar()
        test_period_combo = ttk.Combobox(eval_config_frame, textvariable=test_period_var, 
                                        values=["Last 3 Months", "Last 6 Months", "Last Year"], 
                                        state='readonly', width=30)
        test_period_combo.grid(row=2, column=1, padx=10, pady=5)
        
        evaluate_button = ttk.Button(eval_config_frame, text="Run Evaluation")
        evaluate_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Evaluation results area
        eval_result_frame = ttk.Frame(self.evaluation_frame, padding=10)
        eval_result_frame.pack(pady=10, fill="both", expand=True)
        
        eval_result_label = ttk.Label(eval_result_frame, text="Evaluation Results:")
        eval_result_label.pack(anchor="w")
        
        eval_output = tk.Text(eval_result_frame, height=10, width=80, state="disabled")
        eval_output.pack(pady=5, fill="both", expand=True)
    
    def on_icon_enter(self, event):
        """Handle mouse entering the icon area"""
        # Just change icon, don't automatically show menu
        self.right_icon_label.config(image=self.close_icon)
    
    def on_icon_leave(self, event):
        """Handle mouse leaving the icon area"""
        # If menu is not visible, change back to menu icon
        if not self.menu_visible:
            self.right_icon_label.config(image=self.menu_icon)
    
    def show_dropdown_menu(self):
        """Show the dropdown menu"""
        if not self.menu_visible:
            # Get root window coordinates for proper positioning
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            
            # Calculate position for dropdown menu
            x = self.right_icon_label.winfo_rootx() - root_x
            y = self.top_frame.winfo_height() + self.green_bar.winfo_height()
            
            # Configure dropdown width
            dropdown_width = max(self.icon_width, 200)  # Make sure it's at least 200px wide
            
            # Place dropdown at correct position with proper size
            self.dropdown_frame.place(x=x, y=y, width=dropdown_width)
            
            # Ensure dropdown appears on top
            self.dropdown_frame.lift()
            
            self.menu_visible = True
            
            # Print debug info
            print(f"Dropdown shown at x={x}, y={y}, width={dropdown_width}")
    
    def hide_dropdown_menu(self):
        """Hide the dropdown menu"""
        if self.menu_visible:
            self.dropdown_frame.place_forget()
            self.right_icon_label.config(image=self.menu_icon)
            self.menu_visible = False
            print("Dropdown hidden")
    
    def toggle_dropdown_menu(self, event):
        """Toggle the dropdown menu visibility"""
        if self.menu_visible:
            self.hide_dropdown_menu()
        else:
            self.show_dropdown_menu()
            self.right_icon_label.config(image=self.close_icon)
    
    def close_menu_if_outside(self, event):
        """Close the menu if clicked outside the menu area"""
        if self.menu_visible:
            # Get current widget under mouse
            widget_under_mouse = self.root.winfo_containing(event.x_root, event.y_root)
            
            # Check if click is outside the dropdown and icon
            if widget_under_mouse not in [self.dropdown_frame, self.right_icon_label] and \
               not any(widget_under_mouse == btn for btn in self.dropdown_buttons):
                # Extra check to prevent closing when clicking on menu items
                if not str(widget_under_mouse).startswith(str(self.dropdown_frame)):
                    self.hide_dropdown_menu()
    
    def show_page(self, page_name):
        """Show the selected page and hide others"""
        # Hide all frames first
        if hasattr(self, 'welcome_frame'):
            self.welcome_frame.place_forget()
        
        if hasattr(self, 'train_frame'):
            self.train_frame.pack_forget()
        
        if hasattr(self, 'predict_frame'):
            self.predict_frame.pack_forget()
        
        if hasattr(self, 'evaluation_frame'):
            self.evaluation_frame.pack_forget()
        
        # Show the selected frame
        if page_name == "Welcome":
            self.welcome_frame.place(relx=0.5, rely=0.5, anchor="center")
        elif page_name == "Train":
            self.train_frame.pack(in_=self.content_frame, fill="both", expand=True)
        elif page_name == "Predict":
            self.predict_frame.pack(in_=self.content_frame, fill="both", expand=True)
        elif page_name == "Evaluation":
            self.evaluation_frame.pack(in_=self.content_frame, fill="both", expand=True)
        
        self.current_page = page_name
        print(f"Changed to page: {page_name}")
    
    def handle_menu_selection(self, option):
        """Handle menu option selection"""
        self.hide_dropdown_menu()
        self.show_page(option)
        print(f"Selected option: {option}")
    
    def run_prediction(self):
        selected_model = self.model_var.get()
        selected_target = self.target_var.get()
        selected_duration = self.duration_var.get()
        
        if not selected_model or not selected_target or not selected_duration:
            messagebox.showerror("Error", "Please select all options before running prediction.")
            return
        
        result_text = f"Running {selected_model} for {selected_target} ({selected_duration})...\nPrediction results will be displayed here."
        
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result_text)
        self.output_text.config(state="disabled")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyPredictionGUI(root)
    root.mainloop()