<script setup>
import { computed } from 'vue'

const props = defineProps({
  studios: {
    type: String,
    required: true
  },
  compact: {
    type: Boolean,
    default: false
  }
})

const catalog = {
  cartpole: {
    badge: '01',
    title: 'CartPole',
    hardware: 'CPU',
    duration: '30–60 sec',
    description:
      'Train CartPole with PPO and inspect the reward curve, live console, and policy replay.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole'
  },
  gymnasium: {
    badge: 'LAB',
    title: 'Gymnasium Playground',
    hardware: 'CPU',
    duration: 'seconds to minutes',
    description:
      'Switch among teaching, classic-control, and game environments, then adjust the training recipe.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium'
  },
  vizdoom: {
    badge: '02',
    title: 'ViZDoom',
    hardware: 'CPU',
    duration: '1–4 min',
    description:
      'Train DQN from first-person pixels on aiming, navigation, survival, and combat tasks.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment02-vizdoom'
  },
  atari: {
    badge: '03',
    title: 'Atari / ALE',
    hardware: 'CPU',
    duration: '1–5 min',
    description:
      'Run short-budget DQN experiments on Pong and other pixel games to observe early learning.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari'
  },
  board: {
    badge: '04',
    title: 'Board Games & Self-Play',
    hardware: 'CPU',
    duration: '10–90 sec',
    description:
      'Try CFR, self-play, and policy iteration on compact board games, then inspect complete matches.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay'
  },
  multiagent: {
    badge: '05',
    title: 'Multi-Agent Games',
    hardware: 'CPU',
    duration: '30 sec–3 min',
    description:
      'Train parameter-sharing policies in cooperative and competitive games and inspect agent interactions.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games'
  },
  minigrid: {
    badge: '06',
    title: 'MiniGrid Adventures',
    hardware: 'CPU',
    duration: '20 sec–2 min',
    description:
      'Train exploration policies on rooms, keys, doors, and obstacles, then inspect route replays.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure'
  },
  jax: {
    badge: '07',
    title: 'JAX MinAtar',
    hardware: 'CPU',
    duration: 'first run 30 sec–2 min',
    description:
      'Use JAX compilation and batching on compact arcade tasks and compare cold and warm runs.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games'
  },
  maniskill: {
    badge: '08',
    title: 'ManiSkill Robot Lab',
    hardware: 'xGPU',
    duration: '2–8 min',
    description:
      'Train robot arms to push, pick, stack, and insert objects, with rendered physics replays.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill'
  },
  minestudio: {
    badge: '10',
    title: 'MineStudio Minecraft',
    hardware: 'xGPU',
    duration: 'warm run 3–10 min',
    description:
      'Try navigation, collection, and long-horizon tasks in a visual Minecraft environment.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio'
  },
  unity: {
    badge: '11',
    title: 'Unity ML-Agents',
    hardware: 'xGPU',
    duration: 'Huggy about 4–6 min',
    description:
      'Run Unity scenes such as Huggy fetch, watch sampled live frames, and inspect the final replay GIF.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents'
  },
  ai2thor: {
    badge: '12',
    title: 'AI2-THOR Embodied Home',
    hardware: 'xGPU',
    duration: 'warm run 2–8 min',
    description:
      'Train visual navigation and object-finding policies in homes, then inspect the route taken.',
    href: 'https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied'
  }
}

const copy = {
  eyebrow: 'WALKINGLAB · ONLINE LAB',
  title: 'Want to see the training loop first?',
  description:
    'Open a ready-to-run ModelScope Studio to train in the browser. Watch the curve, live log, and learned-policy preview, then return to the chapter to explain what changed.',
  action: 'Open online training'
}

const selectedStudios = computed(() =>
  props.studios
    .split(',')
    .map((key) => key.trim())
    .filter((key) => catalog[key])
    .map((key) => ({ key, ...catalog[key] }))
)
</script>

<template>
  <aside
    class="online-training"
    :class="{ 'online-training--compact': props.compact }"
    aria-label="Online training resources"
  >
    <div class="online-training__intro">
      <span class="online-training__eyebrow">
        <i aria-hidden="true"></i>
        {{ copy.eyebrow }}
      </span>
      <strong>{{ copy.title }}</strong>
      <p>{{ copy.description }}</p>
    </div>

    <div class="online-training__grid">
      <a
        v-for="studio in selectedStudios"
        :key="studio.key"
        class="online-training__card"
        :href="studio.href"
        target="_blank"
        rel="noopener noreferrer"
      >
        <span class="online-training__badge">{{ studio.badge }}</span>
        <span class="online-training__body">
          <span class="online-training__title">{{ studio.title }}</span>
          <span class="online-training__meta">
            <b>{{ studio.hardware }}</b>
            <span aria-hidden="true">·</span>
            {{ studio.duration }}
          </span>
          <span class="online-training__description">
            {{ studio.description }}
          </span>
          <span class="online-training__action">
            {{ copy.action }}
            <i aria-hidden="true">↗</i>
          </span>
        </span>
      </a>
    </div>
  </aside>
</template>

<style scoped>
.online-training {
  position: relative;
  overflow: hidden;
  margin: 24px 0 30px;
  padding: 22px;
  border: 1px solid
    color-mix(in srgb, var(--vp-c-brand-1) 24%, var(--vp-c-divider));
  border-radius: 22px;
  background:
    radial-gradient(
      circle at 96% 0%,
      color-mix(in srgb, var(--vp-c-brand-1) 16%, transparent) 0,
      transparent 34%
    ),
    linear-gradient(145deg, var(--vp-c-bg-soft) 0%, var(--vp-c-bg) 100%);
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.07);
}

.online-training__intro {
  position: relative;
  max-width: 760px;
}

.online-training__eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 7px;
  color: var(--vp-c-brand-1);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.13em;
}

.online-training__eyebrow i {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #22c55e;
  box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.12);
}

.online-training__intro strong {
  display: block;
  color: var(--vp-c-text-1);
  font-size: 19px;
  line-height: 1.4;
}

.online-training__intro p {
  margin: 7px 0 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.7;
}

.online-training__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.online-training__card {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr);
  gap: 13px;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  color: inherit !important;
  text-decoration: none !important;
  background: color-mix(in srgb, var(--vp-c-bg) 92%, transparent);
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.online-training__card:hover {
  transform: translateY(-2px);
  border-color: color-mix(
    in srgb,
    var(--vp-c-brand-1) 46%,
    var(--vp-c-divider)
  );
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
}

.online-training__badge {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 11px;
  color: #fff;
  font-size: 11px;
  font-weight: 850;
  letter-spacing: 0.04em;
  background: linear-gradient(145deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
}

.online-training__body,
.online-training__title,
.online-training__meta,
.online-training__description,
.online-training__action {
  display: block;
}

.online-training__title {
  color: var(--vp-c-text-1);
  font-size: 15px;
  font-weight: 750;
  line-height: 1.4;
}

.online-training__meta {
  margin-top: 3px;
  color: var(--vp-c-text-3);
  font-size: 12px;
}

.online-training__meta b {
  color: var(--vp-c-brand-1);
  font-weight: 750;
}

.online-training__description {
  margin-top: 8px;
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.55;
}

.online-training__action {
  margin-top: 10px;
  color: var(--vp-c-brand-1);
  font-size: 12px;
  font-weight: 750;
}

.online-training__action i {
  display: inline-block;
  font-style: normal;
  transition: transform 0.2s ease;
}

.online-training__card:hover .online-training__action i {
  transform: translate(2px, -2px);
}

.online-training--compact {
  margin: 18px 0 22px;
  padding: 16px 18px;
  border-radius: 16px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
}

.online-training--compact .online-training__eyebrow {
  margin-bottom: 4px;
}

.online-training--compact .online-training__intro strong {
  font-size: 16px;
}

.online-training--compact .online-training__intro p {
  margin-top: 4px;
  font-size: 13px;
  line-height: 1.6;
}

.online-training--compact .online-training__grid {
  gap: 9px;
  margin-top: 12px;
}

.online-training--compact .online-training__card {
  grid-template-columns: 32px minmax(0, 1fr);
  gap: 11px;
  padding: 12px;
  border-radius: 13px;
}

.online-training--compact .online-training__badge {
  width: 32px;
  height: 32px;
  border-radius: 9px;
  font-size: 10px;
}

.online-training--compact .online-training__title {
  font-size: 14px;
}

.online-training--compact .online-training__description {
  margin-top: 5px;
  font-size: 12px;
}

.online-training--compact .online-training__action {
  margin-top: 7px;
}

.dark .online-training {
  box-shadow: none;
}

@media (max-width: 760px) {
  .online-training {
    padding: 18px;
    border-radius: 18px;
  }

  .online-training__grid {
    grid-template-columns: 1fr;
  }
}
</style>
