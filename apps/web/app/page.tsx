import {
  Camera,
  Brain,
  UserCheck,
  Search,
  FileCheck,
  ChevronDown,
  Folder,
  Database,
  Shield,
  Zap,
  Eye,
  ArrowRight,
  BookOpen,
  Users,
  Globe,
} from "lucide-react";
import { MobileNav } from "@/components/MobileNav";

const workflowSteps = [
  {
    number: "01",
    icon: Camera,
    title: "Upload Sighting",
    description:
      "Conservationists photograph a single elephant during a sighting and upload all images as one folder to Dropbox. A signed-in user imports the folder into the platform, which creates a sighting record and starts analysis.",
  },
  {
    number: "02",
    icon: Brain,
    title: "AI Analysis",
    description:
      "The system processes the entire folder as a unit — combining evidence across all images to generate a draft identification package, including a structured SEEK code and field-by-field predictions.",
  },
  {
    number: "03",
    icon: UserCheck,
    title: "Expert Review",
    description:
      "A human reviewer checks the AI-predicted fields and draft SEEK code, accepting or editing each value. The final reviewed record reflects expert judgment, not blind automation.",
  },
  {
    number: "04",
    icon: Search,
    title: "Candidate Matching",
    description:
      "The reviewed sighting is compared against the full elephant database. The system presents the most likely matches ranked by similarity, giving the reviewer a focused shortlist rather than a silent automatic answer.",
  },
  {
    number: "05",
    icon: FileCheck,
    title: "File Identity",
    description:
      "The reviewer selects an existing elephant identity or creates a new record. The final reviewed SEEK code and structured data are filed into the long-term database with full traceability.",
  },
];

const seekFields = [
  { label: "S", full: "Sex", desc: "Male, female, or unknown" },
  { label: "E", full: "Age", desc: "Estimated age class" },
  { label: "E", full: "Ear — Right", desc: "Right ear features and tears" },
  { label: "K", full: "Ear — Left", desc: "Left ear features and tears" },
  { label: "T", full: "Tusks", desc: "Tusk size, shape, and symmetry" },
  {
    label: "X",
    full: "Extreme Features",
    desc: "Unusual or distinctive marks",
  },
];

const features = [
  {
    icon: Shield,
    title: "Human Oversight Preserved",
    description:
      "Every identification goes through expert review before it is filed. The platform reduces work — it does not remove judgment.",
  },
  {
    icon: Folder,
    title: "Folder-Level Reasoning",
    description:
      "One folder equals one sighting. The system combines evidence across all images in a folder rather than treating each photo in isolation.",
  },
  {
    icon: Zap,
    title: "Faster SEEK Coding",
    description:
      "AI prefills likely values and generates a draft SEEK code. Reviewers validate rather than code from scratch, cutting coding time significantly.",
  },
  {
    icon: Eye,
    title: "Transparent Uncertainty",
    description:
      "Where evidence is incomplete, the system preserves uncertainty rather than guessing. Reviewers always see what the system knows and what it doesn't.",
  },
  {
    icon: Database,
    title: "Scalable Matching",
    description:
      "As the elephant database grows, structured SEEK coding enables consistent, repeatable comparisons rather than subjective visual guessing.",
  },
  {
    icon: Globe,
    title: "Low-Bandwidth Friendly",
    description:
      "Designed for real field conditions. Reviewers work with compressed previews by default; full-resolution images are fetched only when needed.",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-[#faf7f2]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#faf7f2]/90 backdrop-blur-sm border-b border-[#e8dfc8]">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-[#2d5a27] rounded-full flex items-center justify-center">
              <span className="text-white text-xs font-bold">E</span>
            </div>
            <span className="font-semibold text-[#1a3517] text-lg tracking-tight">
              Elephant ID
            </span>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <a
              href="#problem"
              className="text-sm text-[#4a3728] hover:text-[#2d5a27] transition-colors"
            >
              The Problem
            </a>
            <a
              href="#workflow"
              className="text-sm text-[#4a3728] hover:text-[#2d5a27] transition-colors"
            >
              How It Works
            </a>
            <a
              href="#seek"
              className="text-sm text-[#4a3728] hover:text-[#2d5a27] transition-colors"
            >
              SEEK Coding
            </a>
            <a
              href="#features"
              className="text-sm text-[#4a3728] hover:text-[#2d5a27] transition-colors"
            >
              Features
            </a>
          </div>

          <div className="hidden md:flex items-center gap-3">
            <a
              href="#contact"
              className="text-sm text-[#2d5a27] font-medium hover:underline"
            >
              Request Access
            </a>
          </div>

          <MobileNav />
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-24 px-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[#2d5a27]/5 via-transparent to-[#c4961f]/5 pointer-events-none" />
        <div className="max-w-4xl mx-auto text-center relative">
          <div className="inline-flex items-center gap-2 bg-[#2d5a27]/10 border border-[#2d5a27]/20 rounded-full px-4 py-1.5 mb-8">
            <span className="w-1.5 h-1.5 bg-[#2d5a27] rounded-full" />
            <span className="text-xs font-medium text-[#2d5a27] uppercase tracking-wide">
              Conservation Technology
            </span>
          </div>

          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-[#1a3517] leading-tight mb-6">
            Identify elephants.
            <br />
            <span className="text-[#2d5a27]">Faster. Consistently.</span>
            <br />
            With human oversight.
          </h1>

          <p className="text-xl text-[#4a3728] leading-relaxed mb-10 max-w-2xl mx-auto">
            Elephant ID combines AI analysis, structured SEEK coding, and expert
            review to turn a folder of field photographs into a filed identity
            record — reducing repetitive manual work without removing human
            judgment.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="#contact"
              className="inline-flex items-center justify-center gap-2 bg-[#2d5a27] text-white px-8 py-3.5 rounded-lg font-medium hover:bg-[#1a3517] transition-colors"
            >
              Request Early Access
              <ArrowRight size={16} />
            </a>
            <a
              href="#workflow"
              className="inline-flex items-center justify-center gap-2 bg-white border border-[#e8dfc8] text-[#4a3728] px-8 py-3.5 rounded-lg font-medium hover:bg-[#f5f0e8] transition-colors"
            >
              See How It Works
              <ChevronDown size={16} />
            </a>
          </div>

          <div className="mt-16 grid grid-cols-3 gap-8 max-w-lg mx-auto">
            <div className="text-center">
              <div className="text-2xl font-bold text-[#2d5a27]">7</div>
              <div className="text-xs text-[#7a5c48] mt-1">
                SEEK fields coded
              </div>
            </div>
            <div className="text-center border-x border-[#e8dfc8]">
              <div className="text-2xl font-bold text-[#2d5a27]">100%</div>
              <div className="text-xs text-[#7a5c48] mt-1">Human-reviewed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-[#2d5a27]">1 folder</div>
              <div className="text-xs text-[#7a5c48] mt-1">= 1 sighting</div>
            </div>
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section id="problem" className="py-24 px-6 bg-[#1a3517] text-white">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
                The Problem
              </div>
              <h2 className="text-4xl font-bold mb-6 leading-tight">
                Elephant identification matters — but it doesn&apos;t scale.
              </h2>
              <p className="text-[#a8c5a0] leading-relaxed mb-6">
                Knowing which individual elephants are in a population is
                essential for conservation planning, conflict mitigation, and
                longitudinal research. But the current process is{" "}
                <strong className="text-white">
                  labor-intensive, subjective, and difficult to scale.
                </strong>
              </p>
              <p className="text-[#a8c5a0] leading-relaxed">
                Identification relies on photographs that may be partial,
                blurry, or taken from suboptimal angles. No single image
                guarantees all the features needed to tell one elephant from
                another. Experts spend enormous time doing work that could be
                partially automated — if the right tooling existed.
              </p>
            </div>
            <div className="space-y-4">
              {[
                {
                  icon: Users,
                  title: "Expert-dependent",
                  desc: "Identification requires trained field researchers who are in short supply.",
                },
                {
                  icon: BookOpen,
                  title: "Hard to standardize",
                  desc: "Without a consistent coding system, records are difficult to compare across teams.",
                },
                {
                  icon: Database,
                  title: "Difficult to match at scale",
                  desc: "As databases grow, searching by memory or subjective visual comparison breaks down.",
                },
              ].map((item) => (
                <div
                  key={item.title}
                  className="flex gap-4 bg-[#2d5a27]/30 border border-[#2d5a27]/40 rounded-xl p-5"
                >
                  <div className="w-10 h-10 bg-[#2d5a27] rounded-lg flex items-center justify-center shrink-0">
                    <item.icon size={18} className="text-white" />
                  </div>
                  <div>
                    <div className="font-semibold text-white mb-1">
                      {item.title}
                    </div>
                    <div className="text-sm text-[#a8c5a0]">{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="workflow" className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
              End-to-End Workflow
            </div>
            <h2 className="text-4xl font-bold text-[#1a3517] mb-4">
              From field photo to filed identity
            </h2>
            <p className="text-[#7a5c48] text-lg max-w-2xl mx-auto">
              A structured five-step process that takes a folder of sighting
              images and produces a reviewed, filed identification record.
            </p>
          </div>

          <div className="relative">
            <div className="hidden md:block absolute left-8 top-8 bottom-8 w-px bg-[#e8dfc8]" />
            <div className="space-y-4">
              {workflowSteps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <div key={step.number} className="relative flex gap-6 group">
                    <div className="hidden md:flex shrink-0 flex-col items-center">
                      <div
                        className={`w-16 h-16 rounded-2xl flex items-center justify-center z-10 transition-colors ${
                          index % 2 === 0
                            ? "bg-[#2d5a27] text-white"
                            : "bg-[#ede4d2] text-[#2d5a27]"
                        }`}
                      >
                        <Icon size={22} />
                      </div>
                    </div>
                    <div className="flex-1 bg-white border border-[#e8dfc8] rounded-2xl p-6 hover:border-[#2d5a27]/30 hover:shadow-sm transition-all">
                      <div className="flex items-start gap-4">
                        <div className="md:hidden w-10 h-10 bg-[#2d5a27] rounded-xl flex items-center justify-center shrink-0">
                          <Icon size={18} className="text-white" />
                        </div>
                        <div>
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-xs font-mono font-bold text-[#c4961f]">
                              STEP {step.number}
                            </span>
                            <span className="font-semibold text-[#1a3517] text-lg">
                              {step.title}
                            </span>
                          </div>
                          <p className="text-[#7a5c48] leading-relaxed">
                            {step.description}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* SEEK Framework */}
      <section id="seek" className="py-24 px-6 bg-[#ede4d2]">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-start">
            <div>
              <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
                SEEK Coding
              </div>
              <h2 className="text-4xl font-bold text-[#1a3517] mb-6 leading-tight">
                A structured language for elephant identity
              </h2>
              <p className="text-[#4a3728] leading-relaxed mb-6">
                SEEK is the structured identification framework used by Elephant
                ID. It encodes the specific physical features that distinguish
                individual elephants into a consistent, comparable format —
                making matching reliable and reproducible.
              </p>
              <p className="text-[#4a3728] leading-relaxed mb-8">
                The AI drafts a SEEK code from the sighting images. The reviewer
                validates or corrects it. The final filed record stores both the
                structured data and the SEEK code string.
              </p>
              <div className="bg-[#1a3517] rounded-xl px-6 py-4 font-mono text-sm">
                <div className="text-[#c4961f] text-xs mb-2 uppercase tracking-wide">
                  Example SEEK code
                </div>
                <div className="text-[#a8c5a0]">M.AD.T2R1.E2N.E1N.X0.S0</div>
              </div>
            </div>

            <div className="space-y-3">
              {seekFields.map((field, i) => (
                <div
                  key={i}
                  className="flex items-start gap-4 bg-white rounded-xl p-4 border border-[#e8dfc8]"
                >
                  <div className="w-10 h-10 bg-[#2d5a27] rounded-lg flex items-center justify-center shrink-0">
                    <span className="text-white font-bold text-sm font-mono">
                      {field.label}
                    </span>
                  </div>
                  <div>
                    <div className="font-semibold text-[#1a3517] text-sm">
                      {field.full}
                    </div>
                    <div className="text-xs text-[#7a5c48]">{field.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Human-in-the-Loop */}
      <section className="py-24 px-6 bg-[#2d5a27]">
        <div className="max-w-4xl mx-auto text-center">
          <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
            Philosophy
          </div>
          <h2 className="text-4xl font-bold text-white mb-6">
            AI assists. Experts decide.
          </h2>
          <p className="text-xl text-[#a8c5a0] leading-relaxed mb-10 max-w-2xl mx-auto">
            Elephant ID is not designed to replace expert judgment. It is
            designed to reduce the repetitive work that slows experts down.
          </p>
          <div className="grid sm:grid-cols-3 gap-4 text-left">
            {[
              {
                title: "Prefill, don't prescribe",
                desc: "The AI suggests values. Reviewers verify them. The final record reflects human expertise.",
              },
              {
                title: "Preserve uncertainty",
                desc: "When evidence is incomplete, the system flags it — rather than silently guessing a wrong answer.",
              },
              {
                title: "Shortlist, don't decide",
                desc: "Matching surfaces the most likely candidates. The reviewer makes the final identity call.",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="bg-white/10 border border-white/20 rounded-xl p-5"
              >
                <div className="font-semibold text-white mb-2">
                  {item.title}
                </div>
                <div className="text-sm text-[#a8c5a0] leading-relaxed">
                  {item.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
              Platform Features
            </div>
            <h2 className="text-4xl font-bold text-[#1a3517] mb-4">
              Built for real field conditions
            </h2>
            <p className="text-[#7a5c48] text-lg max-w-xl mx-auto">
              Designed to work with the realities of conservation fieldwork —
              variable image quality, slow connections, and non-specialist
              operators.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="bg-white border border-[#e8dfc8] rounded-2xl p-6 hover:border-[#2d5a27]/40 hover:shadow-sm transition-all"
                >
                  <div className="w-11 h-11 bg-[#2d5a27]/10 rounded-xl flex items-center justify-center mb-4">
                    <Icon size={20} className="text-[#2d5a27]" />
                  </div>
                  <h3 className="font-semibold text-[#1a3517] mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-[#7a5c48] leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section
        id="contact"
        className="py-24 px-6 bg-gradient-to-br from-[#1a3517] to-[#2d5a27]"
      >
        <div className="max-w-2xl mx-auto text-center">
          <div className="text-[#c4961f] text-sm font-medium uppercase tracking-widest mb-4">
            Early Access
          </div>
          <h2 className="text-4xl font-bold text-white mb-6">
            Join the early access program
          </h2>
          <p className="text-[#a8c5a0] text-lg leading-relaxed mb-10">
            Elephant ID is in active development. If you work in elephant
            conservation and are interested in piloting the platform with your
            team, reach out to learn more.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <a
              href="mailto:info@elephant-id.org"
              className="inline-flex items-center justify-center gap-2 bg-[#c4961f] text-white px-8 py-3.5 rounded-lg font-medium hover:bg-[#a07a18] transition-colors"
            >
              Get in Touch
              <ArrowRight size={16} />
            </a>
            <a
              href="#workflow"
              className="inline-flex items-center justify-center gap-2 bg-white/10 border border-white/20 text-white px-8 py-3.5 rounded-lg font-medium hover:bg-white/20 transition-colors"
            >
              Learn More
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 bg-[#1a3517] border-t border-[#2d5a27]/40">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-[#2d5a27] rounded-full flex items-center justify-center">
              <span className="text-white text-xs font-bold">E</span>
            </div>
            <span className="text-white font-medium text-sm">Elephant ID</span>
          </div>
          <div className="text-xs text-[#a8c5a0] text-center">
            A human-in-the-loop platform for African elephant identification.
            Built for conservation.
          </div>
          <div className="flex gap-4">
            <a
              href="#problem"
              className="text-xs text-[#a8c5a0] hover:text-white transition-colors"
            >
              About
            </a>
            <a
              href="#contact"
              className="text-xs text-[#a8c5a0] hover:text-white transition-colors"
            >
              Contact
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
