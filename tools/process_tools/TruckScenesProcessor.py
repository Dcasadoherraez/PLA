
class TruckScenesProcessor(ProcessorTemplate):
    def __init__(self, device):
        super(TruckScenesProcessor, self).__init__(device)
        # List all scene directories in 'trainval'
        self.scene_list = sorted([os.path.basename(d) for d in glob.glob(os.path.join(args.dataset_path, 'trainval', 'scene-*'))])

    def process_view_caption(self):
        """Generate captions for each image in the dataset's camera views."""
        captions_view = {}
        print('Processing view captions for TruckScenes dataset...')
        for scene_name in tqdm(self.scene_list):
            scene_path = os.path.join(args.dataset_path, 'trainval', scene_name)
            image_paths = []
            image_names = []
            # Define camera directories
            cameras = ['CAMERA_LEFT_BACK', 'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_BACK', 'CAMERA_RIGHT_FRONT']
            for camera in cameras:
                camera_path = os.path.join(scene_path, 'images', camera)
                # Assuming images are JPEGs; adjust extension if necessary
                for img_file in glob.glob(os.path.join(camera_path, '*.jpg')):
                    image_paths.append(img_file)
                    # Create unique image name with camera prefix
                    img_name = f"{camera}_{os.path.basename(img_file).split('.')[0]}"
                    image_names.append(img_name)
            # Generate captions using the model
            res = self.model.predict_step(image_paths, image_name_list=image_names)
            captions_view[scene_name] = res
        # Save captions to file
        output_file = os.path.join(args.output_dir, f'caption_view_{args.dataset}_{args.caption_model.split("/")[-1]}_{args.tag}.json')
        write_caption_to_file(captions_view, output_file)

    def process_scene_caption(self):
        """Generate scene-level captions by summarizing view captions."""
        print('Processing scene captions for TruckScenes dataset...')
        caption_view_path = args.view_caption_path
        if not caption_view_path:
            raise ValueError("Please provide --view_caption_path to process scene captions.")
        captions_view = json.load(open(caption_view_path, 'r'))
        print(f'Loaded view captions from {caption_view_path}')
        captions_scene = {}
        for scene in tqdm(self.scene_list):
            # Combine all view captions for the scene
            text = '. '.join(captions_view[scene].values())
            # Summarize if longer than 75 words
            if len(text.split(' ')) > 75:
                sum_caption = self.summarizer(text, max_length=75)[0]['summary_text']
            else:
                sum_caption = text
            captions_scene[scene] = sum_caption
        # Save scene captions to file
        output_file = os.path.join(args.output_dir, f'caption_scene_{args.dataset}_{args.caption_model.split("/")[-1]}_{args.tag}.json')
        write_caption_to_file(captions_scene, output_file)
        
    def compute_intersect_and_diff(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = set1 & set2
        old = set1 - set2
        new = set2 - set1
        return old, new, intersection
    
    def process_entity_caption(self):
        print('Processing entity captions for TruckScenes dataset...')
        view_caption_path = self.args.view_caption_path
        if not view_caption_path:
            raise ValueError("Please provide --view_caption_path")
        captions_view = json.load(open(view_caption_path, 'r'))
        captions_entity = {}
        
        for scene in self.scene_list:
            corr_idx_file = os.path.join(self.args.view_caption_corr_idx_path, scene, 'view_caption_corr_idx.pkl')
            if not os.path.exists(corr_idx_file):
                print(f"Correspondence indices not found for {scene}, skipping.")
                continue
            with open(corr_idx_file, 'rb') as f:
                view_caption_corr_idx_scene = pickle.load(f)
            captions_entity[scene] = {}
            entity_num = 0
            
            frames = {}
            for image_name in captions_view[scene].keys():
                if image_name not in view_caption_corr_idx_scene:
                    continue
                frame_idx = view_caption_corr_idx_scene[image_name][0]
                if frame_idx not in frames:
                    frames[frame_idx] = []
                frames[frame_idx].append(image_name)
            
            for frame_idx, image_names in frames.items():
                view_entity_caption_frame = {}
                for img in image_names:
                    caption = captions_view[scene][img]
                    tokens = nltk.word_tokenize(caption)
                    tagged = nltk.pos_tag(tokens)
                    entities = [e[0] for e in tagged if e[1].startswith('NN')]
                    view_entity_caption_frame[img] = ' '.join(entities)
                
                frame_corr_idx = {img: view_caption_corr_idx_scene[img][1] for img in image_names}
                entity_caption_frame = self.get_entity_caption_per_frame(view_entity_caption_frame, frame_corr_idx)
                for caption in entity_caption_frame.values():
                    captions_entity[scene][f'entity_{entity_num}'] = caption
                    entity_num += 1
        
        output_file = os.path.join(self.args.output_dir, f'caption_entity_{self.args.dataset}_{self.args.caption_model.split("/")[-1]}_{self.args.tag}.json')
        with open(output_file, 'w') as f:
            json.dump(captions_entity, f, indent=4)
        print(f"Entity captions saved to {output_file}")
    
    def get_entity_caption_per_frame(self, view_entity_caption, frame_corr_idx):
        entity_caption = {}
        minpoint = 100
        ratio = self.args.entity_overlap_thr
        frame_keys = list(frame_corr_idx.keys())
        for ii in range(len(frame_keys) - 1):
            for jj in range(ii + 1, len(frame_keys)):
                idx1 = np.array(frame_corr_idx[frame_keys[ii]])
                idx2 = np.array(frame_corr_idx[frame_keys[jj]])
                c1 = view_entity_caption[frame_keys[ii]].split(' ')
                c2 = view_entity_caption[frame_keys[jj]].split(' ')
                if 'room' in c1:
                    c1.remove('room')
                if 'room' in c2:
                    c2.remove('room')
                old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
                old_c, new_c, intersection_c = self.compute_intersect_and_diff(c1, c2)
                if len(intersection) > minpoint and len(intersection_c) > 0 and len(intersection) / float(min(len(idx1), len(idx2))) <= ratio:
                    entity_caption[f'entity_{len(entity_caption)}'] = ' '.join(list(intersection_c))
                if len(old) > minpoint and len(old_c) > 0 and len(old) / float(len(idx1)) <= ratio:
                    entity_caption[f'entity_{len(entity_caption)}'] = ' '.join(list(old_c))
                if len(new) > minpoint and len(new_c) > 0 and len(new) / float(len(idx2)) <= ratio:
                    entity_caption[f'entity_{len(entity_caption)}'] = ' '.join(list(new_c))
        return entity_caption

