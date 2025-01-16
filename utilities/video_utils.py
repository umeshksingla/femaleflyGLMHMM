import cv2


def extract_clip_events(video_path, video_start_frame, video_end_frame, output_path, fast=1):
    print(video_path, video_start_frame, video_end_frame, output_path)

    cap = cv2.VideoCapture(video_path)
    print("video_path", video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    for _ in enumerate(range(video_start_frame, video_end_frame, fast)):
        for __ in range(fast):
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > video_end_frame:
                break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
