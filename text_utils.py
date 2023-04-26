import os
import pickle
import numpy as np
import json



def lookup(w2i_lookup, x):
    if x in w2i_lookup:
        return w2i_lookup[x]
    else:
        return len(w2i_lookup)



def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    h = (b-a).astype('timedelta64[h]').astype(int)
    '''if h < -1e-6:
        print(h)
        assert h > 1e-6'''
    return h


class TextReader():
    def __init__(self, dbpath, starttime_path):
        self.dbpath = dbpath
        self.all_files = set(os.listdir(dbpath))
        with open(starttime_path, 'rb') as f:
            self.episodeToStartTime = pickle.load(f)

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id

    def read_all_text_events(self, names):
        texts = {}
        for name in names:
            pid, eid = self.get_name_from_filename(name)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                texts[name] = self.read_text_event_json(text_file_name)
        # for each filename (which contains pateintid, eid) and can be used to merge.
        # it will store a list with timestep and text at that time step.
        return texts
    
    def read_all_text_concat_json(self, names, period_length=48.0):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = ""
                times_array = []
                for (t, txt) in zip(time, texts):
                    if diff(start_time, t) <= period_length + 1e-6:
                        final_concatenated_text = final_concatenated_text + " "+txt
                        times_array.append(t)
                    else:
                        break
            texts_dict[patient_id] = final_concatenated_text
            time_dict[patient_id] = times_array
            start_times[patient_id] = start_time
        return texts_dict, time_dict, start_times
    
    def read_all_text_append_json(self, names, period_length=48.0, NumOfNotes=5, notes_aggeregate='Mean'):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = []
                times_array = []
                for (t, txt) in zip(time, texts):

                    if diff(start_time, t) <= period_length + 1e-6 : #and  diff(start_time, t)>=(-24-1e-6)
                        final_concatenated_text.append(txt)
                        times_array.append(t)
                    else:
                        break
                # if notes_aggeregate == 'First':
                #     texts_dict[patient_id] = final_concatenated_text[:NumOfNotes]
                #     time_dict[patient_id] = times_array[:NumOfNotes]
                # else:
                    # texts_dict[patient_id] = final_concatenated_text[-NumOfNotes:]
                    # time_dict[patient_id] = times_array[-NumOfNotes:]
                if len(final_concatenated_text)==0:
                    continue
                texts_dict[patient_id] = final_concatenated_text
                time_dict[patient_id] = times_array
                start_times[patient_id] = start_time
                # import pdb;
                # pdb.set_trace()
        return texts_dict, time_dict, start_times

    def read_text_event_json(self, text_file_name):
        filepath = os.path.join(self.dbpath, str(text_file_name))
        with open(filepath, 'r') as f:
            d = json.load(f)
        time = sorted(d.keys())
        text = []
        for t in time:
            text.append(" ".join(d[t]))
        assert len(time) == len(text)
        return time, text



def break_indices(indices, size):
    result = []
    start = 0
    lens = []
    while start < len(indices):
        slice_ = indices[start:start + size]
        result.append(slice_)
        lens.append(len(slice_))
        start += size
    return result, max(lens)


def generate_tensor_text(patient_text_list, w2i_lookup, conf_max_len):
    patient_list_of_indices = []
    max_indices_listlen = -1
    max_senteces_listlen = -1

    number_of_docs = []

    for patient_text in patient_text_list:
        # each patient_text is a list of text
        list_of_indices = []
        number_of_docs.append(len(patient_text))
        for sentence in patient_text:
            # each sentence is a list of word
            indices = list(map(lambda x: lookup(
                w2i_lookup, x), str(sentence).split()))
            if conf_max_len > 0:
                indices = indices[:conf_max_len]
            list_of_indices.append(indices)
            max_indices_listlen = max(len(indices), max_indices_listlen)
        patient_list_of_indices.append(list_of_indices)
        max_senteces_listlen = max(len(list_of_indices), max_senteces_listlen)

    pad_token = w2i_lookup['<pad>']

    # 3. 3d pad, padding token.
    # 4. convert to numpy tensor and return
    def extra_pad_tokens(cnt): return [pad_token]*cnt

    padded_patient_list_of_indices = []
    for pt in patient_list_of_indices:
        padded_pt = []
        if len(pt) < max_senteces_listlen:
            pt = pt + [[]]*(max_senteces_listlen-len(pt))
        for l in pt:
            l = l + extra_pad_tokens(max_indices_listlen - len(l))
            padded_pt.append(l)
        padded_patient_list_of_indices.append(padded_pt)

    x = np.array(padded_patient_list_of_indices)
    l = np.array(number_of_docs)

    assert len(x.shape) == 3
    assert x.shape[0] == l.shape[0]
    assert x.shape[0] == len(patient_text_list), "x: {}, l: {}".format(
        str(x.shape), str(len(patient_text_list)))
    return x, l


def assert_shapes(X, mask, output, TimeMask=None, Texts=None):
    assert output.shape[0] == X.shape[0]
    assert mask.shape[0] == X.shape[0]
    assert output.shape[1] == mask.shape[1]
    assert output.shape[1] == X.shape[1]
    if TimeMask is not None:
        assert TimeMask.shape[0] == X.shape[0]
        assert TimeMask.shape[1] == X.shape[1]
        assert Texts.shape[0] == X.shape[0]
        assert Texts.shape[1] <= X.shape[1], "Texts.shape:{}, X.shape: {}".format(
            str(Texts.shape), str(X.shape))
        assert Texts.shape[1] == TimeMask.shape[2], "Texts.shape:{}, TimeMask.shape: {}".format(
            str(Texts.shape), str(TimeMask.shape))


def merge_text_events_with_timeseries(problem_type, data, text_reader, w2i_lookup, conf_max_len,
                                      dump_information=False, fname=None):
    text_not_found = 0
    sucessful = 0

    text_event_lens = []
    data_with_text = []

    if dump_information:
        text_count_by_hour = {}
        patient_count_by_hour = {}
        text_len_by_hour = {}

    maximum_index_output = -1
    for batch in data:
        ip, op, _ = batch['data']
        X = ip[0]
        mask = ip[2]
        if problem_type == 'decom':
            ts = batch['decomp_ts']
            output = op[1]
        elif problem_type == 'los':
            ts = batch['los_ts']
            output = op[2]
            maximum_index_output = max(maximum_index_output, output.max())
        assert_shapes(X, mask, output)
        text_event_dictionary = text_reader.read_all_text_events_json(
            batch['names'])

        max_len = -1
        for i, name in enumerate(batch['names']):
            if name not in text_event_dictionary:
                continue
            text_events = text_event_dictionary[name]
            hours = map(lambda x: x[0], text_events)
            hours = list(filter(lambda h: h <= X.shape[1], hours))
            max_len = max(max_len, len(hours))

        final_items = []
        for i, name in enumerate(batch['names']):
            # timerow represents 1 patient.
            # first timestep is 5.
            if name not in text_event_dictionary:
                text_not_found += 1
                continue
            else:
                sucessful += 1
            # if sucessful % 5000 == 0:
            #    print("Scccessful:", sucessful)
            mask_i = mask[i]
            X_i = X[i]
            output_i = output[i]
            ts_i = ts[i]
            if len(ts_i) == 0:
                continue
            text_events = text_event_dictionary[name]
            assert len(text_events[0]) == 2
            hours = list(map(lambda x: x[0], text_events))[:max_len]
            texts = list(map(lambda x: x[1], text_events))[:max_len]
            if dump_information:
                assert fname is not None
                count = 0
                length = 0
                for t in ts_i:
                    if t in patient_count_by_hour:
                        patient_count_by_hour[t] += 1
                    else:
                        patient_count_by_hour[t] = 1
                    if t in hours:
                        count += 1
                        length += len(texts[hours.index(t)])
                    if t not in text_count_by_hour:
                        text_count_by_hour[t] = 0
                        text_len_by_hour[t] = 0
                    text_count_by_hour[t] += count
                    text_len_by_hour[t] += length

            assert len(hours) == len(texts)
            text_event_lens.append(len(texts))

            # generate 2D TimeMask for 1DConvolution.
            time_mask = np.zeros((mask_i.shape[0], max_len))

            if max(ts_i) >= mask_i.shape[0]:
                ts_i = [ti for ti in ts_i if ti < mask_i.shape[0]]

            for t in ts_i:
                for ind, h in enumerate(hours):
                    if h > t:
                        break
                    time_mask[t][ind] = t-h+1
                    assert time_mask[t][ind] >= 0

            final_items.append(
                {'X': X_i, 'Out': output_i, 'Mask': mask_i, 'Text': texts, 'TimeMask': time_mask})

        if len(final_items) >= 1:
            # Now post process.
            X = np.stack(list(map(lambda x: x['X'], final_items)))
            Output = np.stack(list(map(lambda x: x['Out'], final_items)))
            Mask = np.stack(list(map(lambda x: x['Mask'], final_items)))
            TimeMask = np.stack(
                list(map(lambda x: x['TimeMask'], final_items)))
            Texts, _ = generate_tensor_text(
                list(map(lambda x: x['Text'], final_items)), w2i_lookup, conf_max_len)
            try:
                assert_shapes(X, Mask, Output, TimeMask, Texts)
                data_with_text.append(
                    {'X': X, 'Output': Output, 'Mask': Mask, 'TimeMask': TimeMask, 'Texts': Texts})
            except:
                print("Merge failed due to shape issue")

    print("Text Not found for patients: ", text_not_found)
    print("Sucessful for patients: ", sucessful)
    print("Maximum value in Output: ", maximum_index_output)

    text_event_lens = np.array(text_event_lens)
    from scipy import stats
    print(stats.describe(text_event_lens))

    if dump_information:
        with open(fname, 'wb') as f:
            pickle.dump({'text_count_by_hour': text_count_by_hour,
                         'patient_count_by_hour': patient_count_by_hour,
                         'text_lens_by_hour': text_len_by_hour},
                        f, pickle.HIGHEST_PROTOCOL)

    return data_with_text, text_event_lens
