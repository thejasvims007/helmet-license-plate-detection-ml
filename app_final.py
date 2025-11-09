import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar
from PIL import Image, ImageTk
import cv2
import threading
import os
from ultralytics import YOLO
import torch
import math
import cvzone
from paddleocr import PaddleOCR

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Violation Detection")
        self.root.geometry("1920x1080")

        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        # Using custom trained YOLOv11 model
        self.model = YOLO(r"C:\Users\M Thejasvi\Downloads\Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main\runs\detect\train_yolo11_helmet\weights\best.pt")
        self.classNames = ["with helmet", "without helmet", "rider", "number plate"]
        # Initialize PaddleOCR for OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # GUI Elements
        self.create_widgets()

        # Variables
        self.current_image = None
        self.current_video = None
        self.processing = False
        self.original_image = None  # Store original image for display

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Helmet Violation Detection System", font=("Arial", 20, "bold"))
        title_label.pack(pady=20)

        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Upload Image Button
        self.image_btn = tk.Button(button_frame, text="Upload Image", command=self.upload_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
        self.image_btn.grid(row=0, column=0, padx=10)

        # Upload Video Button
        self.video_btn = tk.Button(button_frame, text="Upload Video", command=self.upload_video, font=("Arial", 12), bg="#2196F3", fg="white", padx=20, pady=10)
        self.video_btn.grid(row=0, column=1, padx=10)

        # Process Button
        self.process_btn = tk.Button(button_frame, text="Process", command=self.process_file, font=("Arial", 12), bg="#FF9800", fg="white", padx=20, pady=10, state=tk.DISABLED)
        self.process_btn.grid(row=0, column=2, padx=10)

        # Canvas frame with scrollbars
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        v_scrollbar = Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        h_scrollbar = Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas for displaying image/video with larger size
        self.canvas = tk.Canvas(canvas_frame, width=1200, height=700, bg="gray",
                               yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        # Results Text Area
        self.results_text = tk.Text(self.root, height=8, width=120, font=("Arial", 10))
        self.results_text.pack(pady=10)

        # Progress Label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.progress_label.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.current_image = file_path
            self.current_video = None
            self.display_image(file_path)
            self.process_btn.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.progress_label.config(text="Image uploaded. Click 'Process' to analyze.")

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.wmv")])
        if file_path:
            self.current_video = file_path
            self.current_image = None
            self.display_video_thumbnail(file_path)
            self.process_btn.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.progress_label.config(text="Video uploaded. Click 'Process' to analyze.")

    def display_image(self, path):
        img = Image.open(path)
        # Store original image size for later use
        self.original_image = img.copy()

        img_width, img_height = img.size
        self.photo = ImageTk.PhotoImage(img)

        # Clear canvas and display at original size, place at top-left, set scrollregion to image size
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, img_width, img_height))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def display_video_thumbnail(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Display at original size
            img_width, img_height = img.size
            self.photo = ImageTk.PhotoImage(img)

            # Place image at top-left and set scrollregion to full image size
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.config(scrollregion=(0, 0, img_width, img_height))
        cap.release()

    def update_canvas_with_frame(self, frame):
        # Clear previous content on canvas
        self.canvas.delete("all")

        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Display at original size
        img_width, img_height = img.size
        self.photo = ImageTk.PhotoImage(img)

        # Place image at top-left and set scrollregion to full image size
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, img_width, img_height))

    def process_file(self):
        if self.processing:
            return

        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Processing...")

        if self.current_image:
            threading.Thread(target=self.process_image).start()
        elif self.current_video:
            threading.Thread(target=self.process_video).start()

    def process_image(self):
        try:
            img = cv2.imread(self.current_image)
            results = self.model(img, stream=True, device=self.device)

            detections = []
            helmet_status = {"with_helmet": 0, "without_helmet": 0, "riders": 0}

            for r in results:
                boxes = r.boxes
                xy = boxes.xyxy
                confidences = boxes.conf
                classes = boxes.cls
                new_boxes = torch.cat((xy.to(self.device), confidences.unsqueeze(1).to(self.device), classes.unsqueeze(1).to(self.device)), 1)

                # Sort boxes by class
                try:
                    new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                except:
                    pass

                rider_boxes = []
                # First, collect all rider boxes
                for i, box in enumerate(new_boxes):
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box[4] * 100)) / 100
                    cls = int(box[5])

                    if cls < len(self.classNames) and self.classNames[cls] == "rider" and conf >= 0.3:
                        rider_boxes.append((x1, y1, x2, y2))

                # Now process all detections
                for i, box in enumerate(new_boxes):
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box[4] * 100)) / 100
                    cls = int(box[5])

                    if cls >= len(self.classNames):
                        continue  # Skip unknown classes

                    class_name = self.classNames[cls]

                    print(f"Detected: {class_name} with conf {conf}")
                    if class_name in ["with helmet", "without helmet", "rider", "number plate"] and conf >= 0.3:
                        detections.append((class_name, conf, (x1, y1, x2, y2)))

                        # Track helmet status
                        if class_name == "with helmet":
                            helmet_status["with_helmet"] += 1
                        elif class_name == "without helmet":
                            helmet_status["without_helmet"] += 1
                        elif class_name == "rider":
                            helmet_status["riders"] += 1

                        # Draw bounding box
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{class_name.upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                           offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

                        if class_name == "number plate":
                            print("Number Plate Detected")
                            crop = img[y1:y1+h, x1:x1+w]
                            try:
                                # Use PaddleOCR for OCR
                                results = self.ocr.ocr(crop, cls=True)
                                if results and results[0]:
                                    texts = [res[1][0] for res in results[0]]
                                    confidences = [res[1][1] for res in results[0]]
                                    text = " ".join(texts)
                                    confidence = sum(confidences) / len(confidences) if confidences else 0
                                    if text and confidence > 0.5:
                                        cvzone.putTextRect(img, f"{text} {round(confidence*100, 1)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                        detections.append((f"Number Plate: {text}", confidence, (x1, y1, x2, y2)))
                                        print(f"Detected Number Plate: {text}")
                                    else:
                                        cvzone.putTextRect(img, "Number Plate Detected (OCR Failed)", (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
                                        detections.append(("Number Plate (OCR failed)", conf, (x1, y1, x2, y2)))
                                else:
                                    cvzone.putTextRect(img, "Number Plate Detected (OCR Failed)", (x1, y1 - 50), scale=1.5, offset=10,
                                                       thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
                                    detections.append(("Number Plate (OCR failed)", conf, (x1, y1, x2, y2)))
                            except Exception as e:
                                print(f"OCR Error: {e}")
                                cvzone.putTextRect(img, "Number Plate Detected (OCR Error)", (x1, y1 - 50), scale=1.5, offset=10,
                                                   thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
                                detections.append(("Number Plate (OCR error)", conf, (x1, y1, x2, y2)))

            # Display processed image at original size
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            img_width, img_height = img_pil.size
            self.photo = ImageTk.PhotoImage(img_pil)

            # Clear canvas and display at original size, place at top-left, set scrollregion to image size
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.config(scrollregion=(0, 0, img_width, img_height))
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Detections in image:\n")
            self.results_text.insert(tk.END, f"Helmet Status:\n")
            self.results_text.insert(tk.END, f"- With helmet: {helmet_status['with_helmet']}\n")
            self.results_text.insert(tk.END, f"- Without helmet: {helmet_status['without_helmet']}\n")
            self.results_text.insert(tk.END, f"- Total riders: {helmet_status['riders']}\n\n")

            for det in detections:
                if isinstance(det[0], str) and det[0].startswith("Number Plate:"):
                    self.results_text.insert(tk.END, f"- {det[0]}: {det[1]*100:.1f}%\n")
                else:
                    self.results_text.insert(tk.END, f"- {det[0]}: {det[1]*100:.1f}%\n")

            self.progress_label.config(text="Image processing complete.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.current_video)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0

            # Create output video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = os.path.join(os.path.dirname(self.current_video), "processed_" + os.path.basename(self.current_video))
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            total_violations = 0
            number_plates = []
            helmet_stats = {"with_helmet": 0, "without_helmet": 0, "riders": 0}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, stream=True, device=self.device)

                for r in results:
                    boxes = r.boxes
                    xy = boxes.xyxy
                    confidences = boxes.conf
                    classes = boxes.cls
                    new_boxes = torch.cat((xy.to(self.device), confidences.unsqueeze(1).to(self.device), classes.unsqueeze(1).to(self.device)), 1)

                    # Sort boxes by class
                    try:
                        new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                    except:
                        pass

                    rider_boxes = []
                    # First, collect all rider boxes
                    for i, box in enumerate(new_boxes):
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box[4] * 100)) / 100
                        cls = int(box[5])

                        if cls < len(self.classNames) and self.classNames[cls] == "rider" and conf >= 0.3:
                            rider_boxes.append((x1, y1, x2, y2))

                    # Now process all detections
                    for i, box in enumerate(new_boxes):
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box[4] * 100)) / 100
                        cls = int(box[5])

                        if cls >= len(self.classNames):
                            continue  # Skip unknown classes

                        class_name = self.classNames[cls]

                        if class_name in ["without helmet", "with helmet", "rider", "number plate"] and conf >= 0.3:
                            if class_name == "without helmet":
                                total_violations += 1
                                helmet_stats["without_helmet"] += 1
                            elif class_name == "with helmet":
                                helmet_stats["with_helmet"] += 1
                            elif class_name == "rider":
                                helmet_stats["riders"] += 1

                            cvzone.cornerRect(frame, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(frame, f"{class_name.upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

                            if class_name == "number plate":
                                crop = frame[y1:y1+h, x1:x1+w]
                                try:
                                    # Use PaddleOCR for OCR
                                    results = self.ocr.ocr(crop, cls=True)
                                    if results and results[0]:
                                        texts = [res[1][0] for res in results[0]]
                                        confidences = [res[1][1] for res in results[0]]
                                        text = " ".join(texts)
                                        confidence = sum(confidences) / len(confidences) if confidences else 0
                                        if text and confidence > 0.5:
                                            cvzone.putTextRect(frame, f"{text} {round(confidence*100, 1)}%",
                                                               (x1, y1 - 50), scale=1.5, offset=10,
                                                               thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                            print(f"Detected Number Plate: {text}")
                                            if text not in number_plates:
                                                number_plates.append(text)
                                        else:
                                            cvzone.putTextRect(frame, "OCR Failed", (x1, y1 - 50), scale=1.5, offset=10,
                                                               thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
                                    else:
                                        cvzone.putTextRect(frame, "OCR Failed", (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
                                except Exception as e:
                                    print(f"OCR Error: {e}")
                                    cvzone.putTextRect(frame, "OCR Error", (x1, y1 - 50), scale=1.5, offset=10,
                                                       thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))

                out.write(frame)
                # Update canvas with processed frame for live display
                self.root.after(0, self.update_canvas_with_frame, frame.copy())
                processed_frames += 1

                # Update progress
                progress = int((processed_frames / frame_count) * 100)
                self.progress_label.config(text=f"Processing video: {progress}% complete")

            cap.release()
            out.release()

            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Video processing complete!\n")
            self.results_text.insert(tk.END, f"Total frames processed: {processed_frames}\n")
            self.results_text.insert(tk.END, f"Helmet Statistics:\n")
            self.results_text.insert(tk.END, f"- With helmet detections: {helmet_stats['with_helmet']}\n")
            self.results_text.insert(tk.END, f"- Without helmet violations: {helmet_stats['without_helmet']}\n")
            self.results_text.insert(tk.END, f"- Total riders detected: {helmet_stats['riders']}\n")
            if number_plates:
                self.results_text.insert(tk.END, f"Number plates detected: {', '.join(number_plates)}\n")
            self.results_text.insert(tk.END, f"Output saved to: {output_path}")

            self.progress_label.config(text="Video processing complete.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.mainloop()
