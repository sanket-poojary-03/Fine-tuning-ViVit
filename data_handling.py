def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224,height=224)
            frames.append(reformatted_frame)
    new=np.stack([x.to_ndarray(format="rgb24") for x in frames])

    return new


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def frames_convert_and_create_dataset_dictionary(directory):
  all_videos=[]
  for i in range(1,11):
      subpath= os.path.join(directory,f'phase{i}')
      video_files = [os.path.join(subpath, file)
                   for file in os.listdir(subpath)
                   if file.lower().endswith('.avi')]
      for j in range(len(video_files)):
        container = av.open(video_files[j])
        indices = sample_frame_indices(clip_len=10, frame_sample_rate=2,seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        all_videos.append({'video': video, 'labels': f"phase{i}"})
  return all_videos
