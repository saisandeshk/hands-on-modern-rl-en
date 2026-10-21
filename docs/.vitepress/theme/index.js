import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import './style.css'
import DpoCodeFocus from '../../chapter14_dpo/components/DpoCodeFocus.vue'
import Layout from './Layout.vue'
import NavCard from './components/NavCard.vue'
import NavGrid from './components/NavGrid.vue'
import OnlineTraining from './components/OnlineTraining.vue'
import PpoCodeFocus from '../../chapter08_ppo/components/PpoCodeFocus.vue'
import GrpoCodeFocus from '../../chapter15_grpo/components/GrpoCodeFocus.vue'
import StepBar from './components/StepBar.vue'

function loadFonts() {
  // Apple style: use native system fonts and load no external fonts.
  // System fonts such as SF Pro are already polished enough.
  if (typeof document === 'undefined') return
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('DpoCodeFocus', DpoCodeFocus)
    ctx.app.component('GrpoCodeFocus', GrpoCodeFocus)
    ctx.app.component('NavCard', NavCard)
    ctx.app.component('NavGrid', NavGrid)
    ctx.app.component('OnlineTraining', OnlineTraining)
    ctx.app.component('PpoCodeFocus', PpoCodeFocus)
    ctx.app.component('StepBar', StepBar)
    loadFonts()
  }
}
