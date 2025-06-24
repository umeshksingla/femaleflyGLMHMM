import os


def clip_session(video_path, interval, output_path):
    import cv2
    print(video_path)
    video_start_frame, video_end_frame = interval
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)   # not sure why it gives 25 for wt_gold files
    print('frame count:', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)
    for _ in enumerate(range(video_start_frame, video_end_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
