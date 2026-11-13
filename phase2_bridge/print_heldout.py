import json
J = json.load(open('carla_eval_heldout/carla_detection_eval.json'))
for t, m in J['models'].items():
    S = m['conf>=0.50']
    print('==', t, '(held-out capture, CARLA, conf>=0.5), frames', S['n_frames'])
    for c in ('ambulance', 'police', 'firetruck', 'car'):
        d = S[c]
        if d['n_obs']:
            print('  %-9s n=%4d tracks=%3d found=%.2f ci=%s correct=%.2f FP/frame=%.3f' % (c, d['n_obs'], d['n_tracks'], d['found'], None if d['found_ci'] is None else tuple(round(x, 2) for x in d['found_ci']), d['correct_class'], S['false_positives_per_frame'][c]))
    print('  emergency pooled:', S['emerg_pooled'])
