# TODO: Enable Live Video Processing Display in GUI

- [x] Add `update_canvas_with_frame(self, frame)` method to HelmetDetectionApp class in app_final.py: Convert OpenCV frame to PIL Image, scale to fit canvas (maintain aspect ratio, no upscaling), center on canvas, and update the canvas image.
- [x] Modify `process_video` method in app_final.py: After drawing detections on the frame, add `self.root.after(0, self.update_canvas_with_frame, frame.copy())` to schedule the canvas update from the thread.
- [x] Test the changes: Run the app, upload a video, process it, and verify live canvas updates during processing.
- [x] Ensure no performance issues due to frequent GUI updates.
