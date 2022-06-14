import numpy as np

def amp_envelop(note_dur, attack_time=0.08, decay_time=0.3, sustain_level=0.6, release_time=0.4, sr=44100):
    if note_dur < attack_time*sr:
        raise ValueError("note duration should be longer than attack time.")
        
    envs = []
    
    env_attack = np.linspace(0,1,int(attack_time*sr))
    envs.append(env_attack)
    if note_dur < (attack_time+release_time)*sr:
        env_release = np.logspace(np.log10(1),np.log10(0.001),int(release_time*sr))[:note_dur-int((attack_time)*sr)]
        envs.append(env_release)
    elif note_dur < (attack_time+decay_time+release_time)*sr:
        env_decay = np.logspace(np.log10(1),np.log10(sustain_level),int(decay_time*sr))[:note_dur-int((attack_time+release_time)*sr)]
        env_release = np.logspace(np.log10(env_decay[-1]),np.log10(0.001),int(release_time*sr))
        envs = envs + [env_decay, env_release]
    else:
        env_decay = np.logspace(np.log10(1),np.log10(sustain_level),int(decay_time*sr))
        env_sustain = np.linspace(sustain_level,sustain_level,note_dur-int((attack_time+decay_time+release_time)*sr))
        env_release = np.logspace(np.log10(sustain_level),np.log10(0.001),int(release_time*sr))
        envs = envs + [env_decay, env_sustain, env_release]
    
    amp_env = np.append(envs[0], envs[1])
    for env in envs[2:]:
        amp_env = np.append(amp_env, env)

    if len(amp_env) < note_dur:
        amp_env = np.pad(amp_env, note_dur-len(amp_env), 'connstant', constant_values=0)

    return amp_env