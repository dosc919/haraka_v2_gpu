
#include "haraka_cuda.h"

#include <stdint.h>


__device__ const uint32_t Te0[256] = { \
	0xa56363c6U, 0x847c7cf8U, 0x997777eeU, 0x8d7b7bf6U,\
	0x0df2f2ffU, 0xbd6b6bd6U, 0xb16f6fdeU, 0x54c5c591U,\
	0x50303060U, 0x03010102U, 0xa96767ceU, 0x7d2b2b56U,\
	0x19fefee7U, 0x62d7d7b5U, 0xe6abab4dU, 0x9a7676ecU,\
	0x45caca8fU, 0x9d82821fU, 0x40c9c989U, 0x877d7dfaU,\
	0x15fafaefU, 0xeb5959b2U, 0xc947478eU, 0x0bf0f0fbU,\
	0xecadad41U, 0x67d4d4b3U, 0xfda2a25fU, 0xeaafaf45U,\
	0xbf9c9c23U, 0xf7a4a453U, 0x967272e4U, 0x5bc0c09bU,\
	0xc2b7b775U, 0x1cfdfde1U, 0xae93933dU, 0x6a26264cU,\
	0x5a36366cU, 0x413f3f7eU, 0x02f7f7f5U, 0x4fcccc83U,\
	0x5c343468U, 0xf4a5a551U, 0x34e5e5d1U, 0x08f1f1f9U,\
	0x937171e2U, 0x73d8d8abU, 0x53313162U, 0x3f15152aU,\
	0x0c040408U, 0x52c7c795U, 0x65232346U, 0x5ec3c39dU,\
	0x28181830U, 0xa1969637U, 0x0f05050aU, 0xb59a9a2fU,\
	0x0907070eU, 0x36121224U, 0x9b80801bU, 0x3de2e2dfU,\
	0x26ebebcdU, 0x6927274eU, 0xcdb2b27fU, 0x9f7575eaU,\
	0x1b090912U, 0x9e83831dU, 0x742c2c58U, 0x2e1a1a34U,\
	0x2d1b1b36U, 0xb26e6edcU, 0xee5a5ab4U, 0xfba0a05bU,\
	0xf65252a4U, 0x4d3b3b76U, 0x61d6d6b7U, 0xceb3b37dU,\
	0x7b292952U, 0x3ee3e3ddU, 0x712f2f5eU, 0x97848413U,\
	0xf55353a6U, 0x68d1d1b9U, 0x00000000U, 0x2cededc1U,\
	0x60202040U, 0x1ffcfce3U, 0xc8b1b179U, 0xed5b5bb6U,\
	0xbe6a6ad4U, 0x46cbcb8dU, 0xd9bebe67U, 0x4b393972U,\
	0xde4a4a94U, 0xd44c4c98U, 0xe85858b0U, 0x4acfcf85U,\
	0x6bd0d0bbU, 0x2aefefc5U, 0xe5aaaa4fU, 0x16fbfbedU,\
	0xc5434386U, 0xd74d4d9aU, 0x55333366U, 0x94858511U,\
	0xcf45458aU, 0x10f9f9e9U, 0x06020204U, 0x817f7ffeU,\
	0xf05050a0U, 0x443c3c78U, 0xba9f9f25U, 0xe3a8a84bU,\
	0xf35151a2U, 0xfea3a35dU, 0xc0404080U, 0x8a8f8f05U,\
	0xad92923fU, 0xbc9d9d21U, 0x48383870U, 0x04f5f5f1U,\
	0xdfbcbc63U, 0xc1b6b677U, 0x75dadaafU, 0x63212142U,\
	0x30101020U, 0x1affffe5U, 0x0ef3f3fdU, 0x6dd2d2bfU,\
	0x4ccdcd81U, 0x140c0c18U, 0x35131326U, 0x2fececc3U,\
	0xe15f5fbeU, 0xa2979735U, 0xcc444488U, 0x3917172eU,\
	0x57c4c493U, 0xf2a7a755U, 0x827e7efcU, 0x473d3d7aU,\
	0xac6464c8U, 0xe75d5dbaU, 0x2b191932U, 0x957373e6U,\
	0xa06060c0U, 0x98818119U, 0xd14f4f9eU, 0x7fdcdca3U,\
	0x66222244U, 0x7e2a2a54U, 0xab90903bU, 0x8388880bU,\
	0xca46468cU, 0x29eeeec7U, 0xd3b8b86bU, 0x3c141428U,\
	0x79dedea7U, 0xe25e5ebcU, 0x1d0b0b16U, 0x76dbdbadU,\
	0x3be0e0dbU, 0x56323264U, 0x4e3a3a74U, 0x1e0a0a14U,\
	0xdb494992U, 0x0a06060cU, 0x6c242448U, 0xe45c5cb8U,\
	0x5dc2c29fU, 0x6ed3d3bdU, 0xefacac43U, 0xa66262c4U,\
	0xa8919139U, 0xa4959531U, 0x37e4e4d3U, 0x8b7979f2U,\
	0x32e7e7d5U, 0x43c8c88bU, 0x5937376eU, 0xb76d6ddaU,\
	0x8c8d8d01U, 0x64d5d5b1U, 0xd24e4e9cU, 0xe0a9a949U,\
	0xb46c6cd8U, 0xfa5656acU, 0x07f4f4f3U, 0x25eaeacfU,\
	0xaf6565caU, 0x8e7a7af4U, 0xe9aeae47U, 0x18080810U,\
	0xd5baba6fU, 0x887878f0U, 0x6f25254aU, 0x722e2e5cU,\
	0x241c1c38U, 0xf1a6a657U, 0xc7b4b473U, 0x51c6c697U,\
	0x23e8e8cbU, 0x7cdddda1U, 0x9c7474e8U, 0x211f1f3eU,\
	0xdd4b4b96U, 0xdcbdbd61U, 0x868b8b0dU, 0x858a8a0fU,\
	0x907070e0U, 0x423e3e7cU, 0xc4b5b571U, 0xaa6666ccU,\
	0xd8484890U, 0x05030306U, 0x01f6f6f7U, 0x120e0e1cU,\
	0xa36161c2U, 0x5f35356aU, 0xf95757aeU, 0xd0b9b969U,\
	0x91868617U, 0x58c1c199U, 0x271d1d3aU, 0xb99e9e27U,\
	0x38e1e1d9U, 0x13f8f8ebU, 0xb398982bU, 0x33111122U,\
	0xbb6969d2U, 0x70d9d9a9U, 0x898e8e07U, 0xa7949433U,\
	0xb69b9b2dU, 0x221e1e3cU, 0x92878715U, 0x20e9e9c9U,\
	0x49cece87U, 0xff5555aaU, 0x78282850U, 0x7adfdfa5U,\
	0x8f8c8c03U, 0xf8a1a159U, 0x80898909U, 0x170d0d1aU,\
	0xdabfbf65U, 0x31e6e6d7U, 0xc6424284U, 0xb86868d0U,\
	0xc3414182U, 0xb0999929U, 0x772d2d5aU, 0x110f0f1eU,\
	0xcbb0b07bU, 0xfc5454a8U, 0xd6bbbb6dU, 0x3a16162cU \
};

 __device__ const uint32_t Te1[256] = { \
	0x6363c6a5U, 0x7c7cf884U, 0x7777ee99U, 0x7b7bf68dU,\
	0xf2f2ff0dU, 0x6b6bd6bdU, 0x6f6fdeb1U, 0xc5c59154U,\
	0x30306050U, 0x01010203U, 0x6767cea9U, 0x2b2b567dU,\
	0xfefee719U, 0xd7d7b562U, 0xabab4de6U, 0x7676ec9aU,\
	0xcaca8f45U, 0x82821f9dU, 0xc9c98940U, 0x7d7dfa87U,\
	0xfafaef15U, 0x5959b2ebU, 0x47478ec9U, 0xf0f0fb0bU,\
	0xadad41ecU, 0xd4d4b367U, 0xa2a25ffdU, 0xafaf45eaU,\
	0x9c9c23bfU, 0xa4a453f7U, 0x7272e496U, 0xc0c09b5bU,\
	0xb7b775c2U, 0xfdfde11cU, 0x93933daeU, 0x26264c6aU,\
	0x36366c5aU, 0x3f3f7e41U, 0xf7f7f502U, 0xcccc834fU,\
	0x3434685cU, 0xa5a551f4U, 0xe5e5d134U, 0xf1f1f908U,\
	0x7171e293U, 0xd8d8ab73U, 0x31316253U, 0x15152a3fU,\
	0x0404080cU, 0xc7c79552U, 0x23234665U, 0xc3c39d5eU,\
	0x18183028U, 0x969637a1U, 0x05050a0fU, 0x9a9a2fb5U,\
	0x07070e09U, 0x12122436U, 0x80801b9bU, 0xe2e2df3dU,\
	0xebebcd26U, 0x27274e69U, 0xb2b27fcdU, 0x7575ea9fU,\
	0x0909121bU, 0x83831d9eU, 0x2c2c5874U, 0x1a1a342eU,\
	0x1b1b362dU, 0x6e6edcb2U, 0x5a5ab4eeU, 0xa0a05bfbU,\
	0x5252a4f6U, 0x3b3b764dU, 0xd6d6b761U, 0xb3b37dceU,\
	0x2929527bU, 0xe3e3dd3eU, 0x2f2f5e71U, 0x84841397U,\
	0x5353a6f5U, 0xd1d1b968U, 0x00000000U, 0xededc12cU,\
	0x20204060U, 0xfcfce31fU, 0xb1b179c8U, 0x5b5bb6edU,\
	0x6a6ad4beU, 0xcbcb8d46U, 0xbebe67d9U, 0x3939724bU,\
	0x4a4a94deU, 0x4c4c98d4U, 0x5858b0e8U, 0xcfcf854aU,\
	0xd0d0bb6bU, 0xefefc52aU, 0xaaaa4fe5U, 0xfbfbed16U,\
	0x434386c5U, 0x4d4d9ad7U, 0x33336655U, 0x85851194U,\
	0x45458acfU, 0xf9f9e910U, 0x02020406U, 0x7f7ffe81U,\
	0x5050a0f0U, 0x3c3c7844U, 0x9f9f25baU, 0xa8a84be3U,\
	0x5151a2f3U, 0xa3a35dfeU, 0x404080c0U, 0x8f8f058aU,\
	0x92923fadU, 0x9d9d21bcU, 0x38387048U, 0xf5f5f104U,\
	0xbcbc63dfU, 0xb6b677c1U, 0xdadaaf75U, 0x21214263U,\
	0x10102030U, 0xffffe51aU, 0xf3f3fd0eU, 0xd2d2bf6dU,\
	0xcdcd814cU, 0x0c0c1814U, 0x13132635U, 0xececc32fU,\
	0x5f5fbee1U, 0x979735a2U, 0x444488ccU, 0x17172e39U,\
	0xc4c49357U, 0xa7a755f2U, 0x7e7efc82U, 0x3d3d7a47U,\
	0x6464c8acU, 0x5d5dbae7U, 0x1919322bU, 0x7373e695U,\
	0x6060c0a0U, 0x81811998U, 0x4f4f9ed1U, 0xdcdca37fU,\
	0x22224466U, 0x2a2a547eU, 0x90903babU, 0x88880b83U,\
	0x46468ccaU, 0xeeeec729U, 0xb8b86bd3U, 0x1414283cU,\
	0xdedea779U, 0x5e5ebce2U, 0x0b0b161dU, 0xdbdbad76U,\
	0xe0e0db3bU, 0x32326456U, 0x3a3a744eU, 0x0a0a141eU,\
	0x494992dbU, 0x06060c0aU, 0x2424486cU, 0x5c5cb8e4U,\
	0xc2c29f5dU, 0xd3d3bd6eU, 0xacac43efU, 0x6262c4a6U,\
	0x919139a8U, 0x959531a4U, 0xe4e4d337U, 0x7979f28bU,\
	0xe7e7d532U, 0xc8c88b43U, 0x37376e59U, 0x6d6ddab7U,\
	0x8d8d018cU, 0xd5d5b164U, 0x4e4e9cd2U, 0xa9a949e0U,\
	0x6c6cd8b4U, 0x5656acfaU, 0xf4f4f307U, 0xeaeacf25U,\
	0x6565caafU, 0x7a7af48eU, 0xaeae47e9U, 0x08081018U,\
	0xbaba6fd5U, 0x7878f088U, 0x25254a6fU, 0x2e2e5c72U,\
	0x1c1c3824U, 0xa6a657f1U, 0xb4b473c7U, 0xc6c69751U,\
	0xe8e8cb23U, 0xdddda17cU, 0x7474e89cU, 0x1f1f3e21U,\
	0x4b4b96ddU, 0xbdbd61dcU, 0x8b8b0d86U, 0x8a8a0f85U,\
	0x7070e090U, 0x3e3e7c42U, 0xb5b571c4U, 0x6666ccaaU,\
	0x484890d8U, 0x03030605U, 0xf6f6f701U, 0x0e0e1c12U,\
	0x6161c2a3U, 0x35356a5fU, 0x5757aef9U, 0xb9b969d0U,\
	0x86861791U, 0xc1c19958U, 0x1d1d3a27U, 0x9e9e27b9U,\
	0xe1e1d938U, 0xf8f8eb13U, 0x98982bb3U, 0x11112233U,\
	0x6969d2bbU, 0xd9d9a970U, 0x8e8e0789U, 0x949433a7U,\
	0x9b9b2db6U, 0x1e1e3c22U, 0x87871592U, 0xe9e9c920U,\
	0xcece8749U, 0x5555aaffU, 0x28285078U, 0xdfdfa57aU,\
	0x8c8c038fU, 0xa1a159f8U, 0x89890980U, 0x0d0d1a17U,\
	0xbfbf65daU, 0xe6e6d731U, 0x424284c6U, 0x6868d0b8U,\
	0x414182c3U, 0x999929b0U, 0x2d2d5a77U, 0x0f0f1e11U,\
	0xb0b07bcbU, 0x5454a8fcU, 0xbbbb6dd6U, 0x16162c3aU \
};

__device__ const uint32_t Te2[256] = { \
	0x63c6a563U, 0x7cf8847cU, 0x77ee9977U, 0x7bf68d7bU,\
	0xf2ff0df2U, 0x6bd6bd6bU, 0x6fdeb16fU, 0xc59154c5U,\
	0x30605030U, 0x01020301U, 0x67cea967U, 0x2b567d2bU,\
	0xfee719feU, 0xd7b562d7U, 0xab4de6abU, 0x76ec9a76U,\
	0xca8f45caU, 0x821f9d82U, 0xc98940c9U, 0x7dfa877dU,\
	0xfaef15faU, 0x59b2eb59U, 0x478ec947U, 0xf0fb0bf0U,\
	0xad41ecadU, 0xd4b367d4U, 0xa25ffda2U, 0xaf45eaafU,\
	0x9c23bf9cU, 0xa453f7a4U, 0x72e49672U, 0xc09b5bc0U,\
	0xb775c2b7U, 0xfde11cfdU, 0x933dae93U, 0x264c6a26U,\
	0x366c5a36U, 0x3f7e413fU, 0xf7f502f7U, 0xcc834fccU,\
	0x34685c34U, 0xa551f4a5U, 0xe5d134e5U, 0xf1f908f1U,\
	0x71e29371U, 0xd8ab73d8U, 0x31625331U, 0x152a3f15U,\
	0x04080c04U, 0xc79552c7U, 0x23466523U, 0xc39d5ec3U,\
	0x18302818U, 0x9637a196U, 0x050a0f05U, 0x9a2fb59aU,\
	0x070e0907U, 0x12243612U, 0x801b9b80U, 0xe2df3de2U,\
	0xebcd26ebU, 0x274e6927U, 0xb27fcdb2U, 0x75ea9f75U,\
	0x09121b09U, 0x831d9e83U, 0x2c58742cU, 0x1a342e1aU,\
	0x1b362d1bU, 0x6edcb26eU, 0x5ab4ee5aU, 0xa05bfba0U,\
	0x52a4f652U, 0x3b764d3bU, 0xd6b761d6U, 0xb37dceb3U,\
	0x29527b29U, 0xe3dd3ee3U, 0x2f5e712fU, 0x84139784U,\
	0x53a6f553U, 0xd1b968d1U, 0x00000000U, 0xedc12cedU,\
	0x20406020U, 0xfce31ffcU, 0xb179c8b1U, 0x5bb6ed5bU,\
	0x6ad4be6aU, 0xcb8d46cbU, 0xbe67d9beU, 0x39724b39U,\
	0x4a94de4aU, 0x4c98d44cU, 0x58b0e858U, 0xcf854acfU,\
	0xd0bb6bd0U, 0xefc52aefU, 0xaa4fe5aaU, 0xfbed16fbU,\
	0x4386c543U, 0x4d9ad74dU, 0x33665533U, 0x85119485U,\
	0x458acf45U, 0xf9e910f9U, 0x02040602U, 0x7ffe817fU,\
	0x50a0f050U, 0x3c78443cU, 0x9f25ba9fU, 0xa84be3a8U,\
	0x51a2f351U, 0xa35dfea3U, 0x4080c040U, 0x8f058a8fU,\
	0x923fad92U, 0x9d21bc9dU, 0x38704838U, 0xf5f104f5U,\
	0xbc63dfbcU, 0xb677c1b6U, 0xdaaf75daU, 0x21426321U,\
	0x10203010U, 0xffe51affU, 0xf3fd0ef3U, 0xd2bf6dd2U,\
	0xcd814ccdU, 0x0c18140cU, 0x13263513U, 0xecc32fecU,\
	0x5fbee15fU, 0x9735a297U, 0x4488cc44U, 0x172e3917U,\
	0xc49357c4U, 0xa755f2a7U, 0x7efc827eU, 0x3d7a473dU,\
	0x64c8ac64U, 0x5dbae75dU, 0x19322b19U, 0x73e69573U,\
	0x60c0a060U, 0x81199881U, 0x4f9ed14fU, 0xdca37fdcU,\
	0x22446622U, 0x2a547e2aU, 0x903bab90U, 0x880b8388U,\
	0x468cca46U, 0xeec729eeU, 0xb86bd3b8U, 0x14283c14U,\
	0xdea779deU, 0x5ebce25eU, 0x0b161d0bU, 0xdbad76dbU,\
	0xe0db3be0U, 0x32645632U, 0x3a744e3aU, 0x0a141e0aU,\
	0x4992db49U, 0x060c0a06U, 0x24486c24U, 0x5cb8e45cU,\
	0xc29f5dc2U, 0xd3bd6ed3U, 0xac43efacU, 0x62c4a662U,\
	0x9139a891U, 0x9531a495U, 0xe4d337e4U, 0x79f28b79U,\
	0xe7d532e7U, 0xc88b43c8U, 0x376e5937U, 0x6ddab76dU,\
	0x8d018c8dU, 0xd5b164d5U, 0x4e9cd24eU, 0xa949e0a9U,\
	0x6cd8b46cU, 0x56acfa56U, 0xf4f307f4U, 0xeacf25eaU,\
	0x65caaf65U, 0x7af48e7aU, 0xae47e9aeU, 0x08101808U,\
	0xba6fd5baU, 0x78f08878U, 0x254a6f25U, 0x2e5c722eU,\
	0x1c38241cU, 0xa657f1a6U, 0xb473c7b4U, 0xc69751c6U,\
	0xe8cb23e8U, 0xdda17cddU, 0x74e89c74U, 0x1f3e211fU,\
	0x4b96dd4bU, 0xbd61dcbdU, 0x8b0d868bU, 0x8a0f858aU,\
	0x70e09070U, 0x3e7c423eU, 0xb571c4b5U, 0x66ccaa66U,\
	0x4890d848U, 0x03060503U, 0xf6f701f6U, 0x0e1c120eU,\
	0x61c2a361U, 0x356a5f35U, 0x57aef957U, 0xb969d0b9U,\
	0x86179186U, 0xc19958c1U, 0x1d3a271dU, 0x9e27b99eU,\
	0xe1d938e1U, 0xf8eb13f8U, 0x982bb398U, 0x11223311U,\
	0x69d2bb69U, 0xd9a970d9U, 0x8e07898eU, 0x9433a794U,\
	0x9b2db69bU, 0x1e3c221eU, 0x87159287U, 0xe9c920e9U,\
	0xce8749ceU, 0x55aaff55U, 0x28507828U, 0xdfa57adfU,\
	0x8c038f8cU, 0xa159f8a1U, 0x89098089U, 0x0d1a170dU,\
	0xbf65dabfU, 0xe6d731e6U, 0x4284c642U, 0x68d0b868U,\
	0x4182c341U, 0x9929b099U, 0x2d5a772dU, 0x0f1e110fU,\
	0xb07bcbb0U, 0x54a8fc54U, 0xbb6dd6bbU, 0x162c3a16U \
};

__device__ const uint32_t Te3[256] = { \
	0xc6a56363U, 0xf8847c7cU, 0xee997777U, 0xf68d7b7bU,\
	0xff0df2f2U, 0xd6bd6b6bU, 0xdeb16f6fU, 0x9154c5c5U,\
	0x60503030U, 0x02030101U, 0xcea96767U, 0x567d2b2bU,\
	0xe719fefeU, 0xb562d7d7U, 0x4de6ababU, 0xec9a7676U,\
	0x8f45cacaU, 0x1f9d8282U, 0x8940c9c9U, 0xfa877d7dU,\
	0xef15fafaU, 0xb2eb5959U, 0x8ec94747U, 0xfb0bf0f0U,\
	0x41ecadadU, 0xb367d4d4U, 0x5ffda2a2U, 0x45eaafafU,\
	0x23bf9c9cU, 0x53f7a4a4U, 0xe4967272U, 0x9b5bc0c0U,\
	0x75c2b7b7U, 0xe11cfdfdU, 0x3dae9393U, 0x4c6a2626U,\
	0x6c5a3636U, 0x7e413f3fU, 0xf502f7f7U, 0x834fccccU,\
	0x685c3434U, 0x51f4a5a5U, 0xd134e5e5U, 0xf908f1f1U,\
	0xe2937171U, 0xab73d8d8U, 0x62533131U, 0x2a3f1515U,\
	0x080c0404U, 0x9552c7c7U, 0x46652323U, 0x9d5ec3c3U,\
	0x30281818U, 0x37a19696U, 0x0a0f0505U, 0x2fb59a9aU,\
	0x0e090707U, 0x24361212U, 0x1b9b8080U, 0xdf3de2e2U,\
	0xcd26ebebU, 0x4e692727U, 0x7fcdb2b2U, 0xea9f7575U,\
	0x121b0909U, 0x1d9e8383U, 0x58742c2cU, 0x342e1a1aU,\
	0x362d1b1bU, 0xdcb26e6eU, 0xb4ee5a5aU, 0x5bfba0a0U,\
	0xa4f65252U, 0x764d3b3bU, 0xb761d6d6U, 0x7dceb3b3U,\
	0x527b2929U, 0xdd3ee3e3U, 0x5e712f2fU, 0x13978484U,\
	0xa6f55353U, 0xb968d1d1U, 0x00000000U, 0xc12cededU,\
	0x40602020U, 0xe31ffcfcU, 0x79c8b1b1U, 0xb6ed5b5bU,\
	0xd4be6a6aU, 0x8d46cbcbU, 0x67d9bebeU, 0x724b3939U,\
	0x94de4a4aU, 0x98d44c4cU, 0xb0e85858U, 0x854acfcfU,\
	0xbb6bd0d0U, 0xc52aefefU, 0x4fe5aaaaU, 0xed16fbfbU,\
	0x86c54343U, 0x9ad74d4dU, 0x66553333U, 0x11948585U,\
	0x8acf4545U, 0xe910f9f9U, 0x04060202U, 0xfe817f7fU,\
	0xa0f05050U, 0x78443c3cU, 0x25ba9f9fU, 0x4be3a8a8U,\
	0xa2f35151U, 0x5dfea3a3U, 0x80c04040U, 0x058a8f8fU,\
	0x3fad9292U, 0x21bc9d9dU, 0x70483838U, 0xf104f5f5U,\
	0x63dfbcbcU, 0x77c1b6b6U, 0xaf75dadaU, 0x42632121U,\
	0x20301010U, 0xe51affffU, 0xfd0ef3f3U, 0xbf6dd2d2U,\
	0x814ccdcdU, 0x18140c0cU, 0x26351313U, 0xc32fececU,\
	0xbee15f5fU, 0x35a29797U, 0x88cc4444U, 0x2e391717U,\
	0x9357c4c4U, 0x55f2a7a7U, 0xfc827e7eU, 0x7a473d3dU,\
	0xc8ac6464U, 0xbae75d5dU, 0x322b1919U, 0xe6957373U,\
	0xc0a06060U, 0x19988181U, 0x9ed14f4fU, 0xa37fdcdcU,\
	0x44662222U, 0x547e2a2aU, 0x3bab9090U, 0x0b838888U,\
	0x8cca4646U, 0xc729eeeeU, 0x6bd3b8b8U, 0x283c1414U,\
	0xa779dedeU, 0xbce25e5eU, 0x161d0b0bU, 0xad76dbdbU,\
	0xdb3be0e0U, 0x64563232U, 0x744e3a3aU, 0x141e0a0aU,\
	0x92db4949U, 0x0c0a0606U, 0x486c2424U, 0xb8e45c5cU,\
	0x9f5dc2c2U, 0xbd6ed3d3U, 0x43efacacU, 0xc4a66262U,\
	0x39a89191U, 0x31a49595U, 0xd337e4e4U, 0xf28b7979U,\
	0xd532e7e7U, 0x8b43c8c8U, 0x6e593737U, 0xdab76d6dU,\
	0x018c8d8dU, 0xb164d5d5U, 0x9cd24e4eU, 0x49e0a9a9U,\
	0xd8b46c6cU, 0xacfa5656U, 0xf307f4f4U, 0xcf25eaeaU,\
	0xcaaf6565U, 0xf48e7a7aU, 0x47e9aeaeU, 0x10180808U,\
	0x6fd5babaU, 0xf0887878U, 0x4a6f2525U, 0x5c722e2eU,\
	0x38241c1cU, 0x57f1a6a6U, 0x73c7b4b4U, 0x9751c6c6U,\
	0xcb23e8e8U, 0xa17cddddU, 0xe89c7474U, 0x3e211f1fU,\
	0x96dd4b4bU, 0x61dcbdbdU, 0x0d868b8bU, 0x0f858a8aU,\
	0xe0907070U, 0x7c423e3eU, 0x71c4b5b5U, 0xccaa6666U,\
	0x90d84848U, 0x06050303U, 0xf701f6f6U, 0x1c120e0eU,\
	0xc2a36161U, 0x6a5f3535U, 0xaef95757U, 0x69d0b9b9U,\
	0x17918686U, 0x9958c1c1U, 0x3a271d1dU, 0x27b99e9eU,\
	0xd938e1e1U, 0xeb13f8f8U, 0x2bb39898U, 0x22331111U,\
	0xd2bb6969U, 0xa970d9d9U, 0x07898e8eU, 0x33a79494U,\
	0x2db69b9bU, 0x3c221e1eU, 0x15928787U, 0xc920e9e9U,\
	0x8749ceceU, 0xaaff5555U, 0x50782828U, 0xa57adfdfU,\
	0x038f8c8cU, 0x59f8a1a1U, 0x09808989U, 0x1a170d0dU,\
	0x65dabfbfU, 0xd731e6e6U, 0x84c64242U, 0xd0b86868U,\
	0x82c34141U, 0x29b09999U, 0x5a772d2dU, 0x1e110f0fU,\
	0x7bcbb0b0U, 0xa8fc5454U, 0x6dd6bbbbU, 0x2c3a1616U \
};

__constant__ const uint32_t RC[256] = {\
	0x0684704c, 0xe620c00a, 0xb2c5fef0, 0x75817b9d, \
	0x8b66b4e1, 0x88f3a06b, 0x640f6ba4, 0x2f08f717, \
	0x3402de2d, 0x53f28498, 0xcf029d60, 0x9f029114, \
	0x0ed6eae6, 0x2e7b4f08, 0xbbf3bcaf, 0xfd5b4f79, \
	0xcbcfb0cb, 0x4872448b, 0x79eecd1c, 0xbe397044, \
	0x7eeacdee, 0x6e9032b7, 0x8d5335ed, 0x2b8a057b, \
	0x67c28f43, 0x5e2e7cd0, 0xe2412761, 0xda4fef1b, \
	0x2924d9b0, 0xafcacc07, 0x675ffde2, 0x1fc70b3b, \
	0xab4d63f1, 0xe6867fe9, 0xecdb8fca, 0xb9d465ee, \
	0x1c30bf84, 0xd4b7cd64, 0x5b2a404f, 0xad037e33, \
	0xb2cc0bb9, 0x941723bf, 0x69028b2e, 0x8df69800, \
	0xfa0478a6, 0xde6f5572, 0x4aaa9ec8, 0x5c9d2d8a, \
	0xdfb49f2b, 0x6b772a12, 0x0efa4f2e, 0x29129fd4, \
	0x1ea10344, 0xf449a236, 0x32d611ae, 0xbb6a12ee, \
	0xaf044988, 0x4b050084, 0x5f9600c9, 0x9ca8eca6, \
	0x21025ed8, 0x9d199c4f, 0x78a2c7e3, 0x27e593ec, \
	0xbf3aaaf8, 0xa759c9b7, 0xb9282ecd, 0x82d40173, \
	0x6260700d, 0x6186b017, 0x37f2efd9, 0x10307d6b, \
	0x5aca45c2, 0x21300443, 0x81c29153, 0xf6fc9ac6, \
	0x9223973c, 0x226b68bb, 0x2caf92e8, 0x36d1943a, \
	0xd3bf9238, 0x225886eb, 0x6cbab958, 0xe51071b4, \
	0xdb863ce5, 0xaef0c677, 0x933dfddd, 0x24e1128d, \
	0xbb606268, 0xffeba09c, 0x83e48de3, 0xcb2212b1, \
	0x734bd3dc, 0xe2e4d19c, 0x2db91a4e, 0xc72bf77d, \
	0x43bb47c3, 0x61301b43, 0x4b1415c4, 0x2cb3924e, \
	0xdba775a8, 0xe707eff6, 0x03b231dd, 0x16eb6899, \
	0x6df3614b, 0x3c755977, 0x8e5e2302, 0x7eca472c, \
	0xcda75a17, 0xd6de7d77, 0x6d1be5b9, 0xb88617f9, \
	0xec6b43f0, 0x6ba8e9aa, 0x9d6c069d, 0xa946ee5d, \
	0xcb1e6950, 0xf957332b, 0xa2531159, 0x3bf327c1, \
	0x2cee0c75, 0x00da619c, 0xe4ed0353, 0x600ed0d9, \
	0xf0b1a5a1, 0x96e90cab, 0x80bbbabc, 0x63a4a350, \
	0xae3db102, 0x5e962988, 0xab0dde30, 0x938dca39, \
	0x17bb8f38, 0xd554a40b, 0x8814f3a8, 0x2e75b442, \
	0x34bb8a5b, 0x5f427fd7, 0xaeb6b779, 0x360a16f6, \
	0x26f65241, 0xcbe55438, 0x43ce5918, 0xffbaafde, \
	0x4ce99a54, 0xb9f3026a, 0xa2ca9cf7, 0x839ec978, \
	0xae51a51a, 0x1bdff7be, 0x40c06e28, 0x22901235, \
	0xa0c1613c, 0xba7ed22b, 0xc173bc0f, 0x48a659cf, \
	0x756acc03, 0x02288288, 0x4ad6bdfd, 0xe9c59da1, \
 };


#define AES_ENC_ROUND(n,D,S) \
		AES_ENC_STEP(n,D,S,0,1,2,3); \
		AES_ENC_STEP(n,D,S,1,2,3,0); \
		AES_ENC_STEP(n,D,S,2,3,0,1); \
		AES_ENC_STEP(n,D,S,3,0,1,2);

#define AES_ENC_STEP(n,D,S,W,X,Y,Z) \
		D##W  = TE(0)[ S##W        & 0xff]; \
		D##W ^= TE(1)[(S##X >>  8) & 0xff]; \
		D##W ^= TE(2)[(S##Y >> 16) & 0xff]; \
		D##W ^= TE(3)[ S##Z >> 24        ]; \
		D##W ^= RC[n + 3 - W];


#define TX (__umul24(blockIdx.x,blockDim.x) + threadIdx.x)
#define SX (threadIdx.x)


#define TE(n)	Tes##n
#define COPY_CONSTANT_SHARED_ENC\
		__shared__ uint32_t Tes0[256];\
		__shared__ uint32_t Tes1[256];\
		__shared__ uint32_t Tes2[256];\
		__shared__ uint32_t Tes3[256];\
		Tes0[SX] = Te0[SX];\
		Tes1[SX] = Te1[SX];\
		Tes2[SX] = Te2[SX];\
		Tes3[SX] = Te3[SX];\
		__syncthreads();


#define GLOBAL_LOAD_SHARED_SETUP_512 \
		register uint32_t r0, r1, r2, r3; \
		register uint32_t s0, s1, s2, s3; \
		register uint32_t t0, t1, t2, t3; \
		register uint32_t u0, u1, u2, u3; \
		register uint32_t v0, v1, v2, v3; \
		register uint64_t load[2]; \
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(msg)[4 * TX]; \
		s0 = load[0]; \
		s1 = load[0] >> 32; \
		s2 = load[1]; \
		s3 = load[1] >> 32; \
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(msg)[4 * TX + 1]; \
		t0 = load[0]; \
		t1 = load[0] >> 32; \
		t2 = load[1]; \
		t3 = load[1] >> 32;\
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(msg)[4 * TX + 2]; \
		u0 = load[0]; \
		u1 = load[0] >> 32; \
		u2 = load[1]; \
		u3 = load[1] >> 32;\
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(msg)[4 * TX + 3]; \
		v0 = load[0]; \
		v1 = load[0] >> 32; \
		v2 = load[1]; \
		v3 = load[1] >> 32;


#define MIX_512 \
	r0 = s0; \
	r1 = u2; \
\
	s0 = s3; \
	s3 = v3; \
	v3 = v2; \
	v2 = t2; \
	t2 = v0; \
	v0 = s2; \
	s2 = t3; \
	t3 = t0; \
	t0 = u0; \
	u0 = u1; \
	u1 = s1; \
	s1 = u3; \
	u3 = t1; \
	t1 = r0; \
\
	u2 = v1; \
	v1 = r1;
	

__global__ void haraka512Kernel(const uint64_t* msg, uint64_t* hash, const uint32_t num_msgs)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_msgs))
		return;

	GLOBAL_LOAD_SHARED_SETUP_512

	AES_ENC_ROUND(0, r, s);
	AES_ENC_ROUND(16, s, r);

	AES_ENC_ROUND(1 * 4, r, t);
	AES_ENC_ROUND(1 * 4 + 16, t, r);

	AES_ENC_ROUND(2 * 4, r, u);
	AES_ENC_ROUND(2 * 4 + 16, u, r);

	AES_ENC_ROUND(3 * 4, r, v);
	AES_ENC_ROUND(3 * 4 + 16, v, r);

	MIX_512

	AES_ENC_ROUND(0 * 4 + 32, r, s);
	AES_ENC_ROUND(0 * 4 + 48, s, r);

	AES_ENC_ROUND(1 * 4 + 32, r, t);
	AES_ENC_ROUND(1 * 4 + 48, t, r);

	AES_ENC_ROUND(2 * 4 + 32, r, u);
	AES_ENC_ROUND(2 * 4 + 48, u, r);

	AES_ENC_ROUND(3 * 4 + 32, r, v);
	AES_ENC_ROUND(3 * 4 + 48, v, r);

	MIX_512

	AES_ENC_ROUND(0 * 4 + 64, r, s);
	AES_ENC_ROUND(0 * 4 + 80, s, r);

	AES_ENC_ROUND(1 * 4 + 64, r, t);
	AES_ENC_ROUND(1 * 4 + 80, t, r);

	AES_ENC_ROUND(2 * 4 + 64, r, u);
	AES_ENC_ROUND(2 * 4 + 80, u, r);

	AES_ENC_ROUND(3 * 4 + 64, r, v);
	AES_ENC_ROUND(3 * 4 + 80, v, r);

	MIX_512

	AES_ENC_ROUND(0 * 4 + 96, r, s);
	AES_ENC_ROUND(0 * 4 + 112, s, r);

	AES_ENC_ROUND(1 * 4 + 96, r, t);
	AES_ENC_ROUND(1 * 4 + 112, t, r);

	AES_ENC_ROUND(2 * 4 + 96, r, u);
	AES_ENC_ROUND(2 * 4 + 112, u, r);

	AES_ENC_ROUND(3 * 4 + 96, r, v);
	AES_ENC_ROUND(3 * 4 + 112, v, r);

	MIX_512

	AES_ENC_ROUND(0 * 4 + 128, r, s);
	AES_ENC_ROUND(0 * 4 + 144, s, r);

	AES_ENC_ROUND(1 * 4 + 128, r, t);
	AES_ENC_ROUND(1 * 4 + 144, t, r);

	AES_ENC_ROUND(2 * 4 + 128, r, u);
	AES_ENC_ROUND(2 * 4 + 144, u, r);

	AES_ENC_ROUND(3 * 4 + 128, r, v);
	AES_ENC_ROUND(3 * 4 + 144, v, r);

	MIX_512

	load[1] = s2 | ((uint64_t)s3) << 32;
	hash[TX * 4] = load[1] ^ msg[8 * TX + 1];

	load[1] = t2 | ((uint64_t)t3) << 32;
	hash[TX * 4 + 1] = load[1] ^ msg[8 * TX + 3];

	load[1] = u0 | ((uint64_t)u1) << 32;
	hash[TX * 4 + 2] = load[1] ^ msg[8 * TX + 4];

	load[1] = v0 | ((uint64_t)v1) << 32;
	load[1] ^= load[0];

	hash[TX * 4 + 3] = load[1];
}

#define GLOBAL_LOAD_SHARED_SETUP_256(input) \
		register uint32_t r0, r1, r2, r3; \
		register uint32_t s0, s1, s2, s3; \
		register uint32_t t0, t1, t2, t3; \
		register uint64_t load[4]; \
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(input)[2 * TX]; \
		s0 = load[0]; \
		s1 = load[0] >> 32; \
		s2 = load[1]; \
		s3 = load[1] >> 32; \
		reinterpret_cast<uint4*>(load)[1] = reinterpret_cast<const uint4*>(input)[2 * TX + 1]; \
		t0 = load[2]; \
		t1 = load[2] >> 32; \
		t2 = load[3]; \
		t3 = load[3] >> 32;


#define MIX_256 \
	r0 = s1; \
	r1 = s3; \
\
	s1 = t0; \
	t0 = s2; \
	s2 = r0; \
\
	s3 = t1; \
	t1 = t2; \
	t2 = r1;

__global__ void haraka256Kernel(const uint64_t* msg, uint64_t* hash, const uint32_t num_msgs)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_msgs))
		return;

	GLOBAL_LOAD_SHARED_SETUP_256(msg)

	AES_ENC_ROUND(0 * 4, r, s);
	AES_ENC_ROUND(0 * 4 + 8, s, r);

	AES_ENC_ROUND(1 * 4, r, t);
	AES_ENC_ROUND(1 * 4 + 8, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 16, r, s);
	AES_ENC_ROUND(0 * 4 + 24, s, r);

	AES_ENC_ROUND(1 * 4 + 16, r, t);
	AES_ENC_ROUND(1 * 4 + 24, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 32, r, s);
	AES_ENC_ROUND(0 * 4 + 40, s, r);

	AES_ENC_ROUND(1 * 4 + 32, r, t);
	AES_ENC_ROUND(1 * 4 + 40, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 48, r, s);
	AES_ENC_ROUND(0 * 4 + 56, s, r);

	AES_ENC_ROUND(1 * 4 + 48, r, t);
	AES_ENC_ROUND(1 * 4 + 56, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 64, r, s);
	AES_ENC_ROUND(0 * 4 + 72, s, r);

	AES_ENC_ROUND(1 * 4 + 64, r, t);
	AES_ENC_ROUND(1 * 4 + 72, t, r);

	MIX_256

	load[0] ^= (s0 | ((uint64_t)s1) << 32);
	load[1] ^= (s2 | ((uint64_t)s3) << 32);
	reinterpret_cast<uint4*>(hash)[2 * TX] = reinterpret_cast<uint4*>(load)[0];

	load[2] ^= (t0 | ((uint64_t)t1) << 32);
	load[3] ^= (t2 | ((uint64_t)t3) << 32);
	reinterpret_cast<uint4*>(hash)[2 * TX + 1] = reinterpret_cast<uint4*>(load)[1];
}

__global__ void harakaOTSKernel(const uint64_t* msg, uint16_t* b, const uint32_t num_msgs)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_msgs))
		return;

	GLOBAL_LOAD_SHARED_SETUP_256(msg)

	AES_ENC_ROUND(0 * 4, r, s);
	AES_ENC_ROUND(0 * 4 + 8, s, r);

	AES_ENC_ROUND(1 * 4, r, t);
	AES_ENC_ROUND(1 * 4 + 8, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 16, r, s);
	AES_ENC_ROUND(0 * 4 + 24, s, r);

	AES_ENC_ROUND(1 * 4 + 16, r, t);
	AES_ENC_ROUND(1 * 4 + 24, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 32, r, s);
	AES_ENC_ROUND(0 * 4 + 40, s, r);

	AES_ENC_ROUND(1 * 4 + 32, r, t);
	AES_ENC_ROUND(1 * 4 + 40, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 48, r, s);
	AES_ENC_ROUND(0 * 4 + 56, s, r);

	AES_ENC_ROUND(1 * 4 + 48, r, t);
	AES_ENC_ROUND(1 * 4 + 56, t, r);

	MIX_256

	AES_ENC_ROUND(0 * 4 + 64, r, s);
	AES_ENC_ROUND(0 * 4 + 72, s, r);

	AES_ENC_ROUND(1 * 4 + 64, r, t);
	AES_ENC_ROUND(1 * 4 + 72, t, r);

	MIX_256

	load[0] ^= (s0 | ((uint64_t)s1) << 32);
	load[1] ^= (s2 | ((uint64_t)s3) << 32);

	load[2] ^= (t0 | ((uint64_t)t1) << 32);
	load[3] ^= (t2 | ((uint64_t)t3) << 32);

#pragma unroll
	for(int i = 1; i < 17; ++i)
		b[17 * TX + i] = reinterpret_cast<uint16_t*>(load)[i - 1];

	uint16_t checksum = 256 - reinterpret_cast<uint8_t*>(load)[0];

#pragma unroll
	for (int i = 1; i < 32; ++i)
		checksum += (256 - reinterpret_cast<uint8_t*>(load)[i]);

	b[17 * TX] = checksum;
}

__global__ void harakaOTSCreateSignatureKernel(const uint64_t* priv_key, uint64_t* pub_key, uint64_t* signature, const uint8_t* b, const uint32_t num_chunks)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_chunks))
		return;

	GLOBAL_LOAD_SHARED_SETUP_256(priv_key)

	uint8_t b_ = b[TX];

#pragma unroll
	for (int i = 0; i < 255; ++i)
	{
		if (i == b_)
		{
			reinterpret_cast<uint4*>(signature)[2 * TX] = reinterpret_cast<uint4*>(load)[0];
			reinterpret_cast<uint4*>(signature)[2 * TX + 1] = reinterpret_cast<uint4*>(load)[1];
		}

		AES_ENC_ROUND(0 * 4, r, s);
		AES_ENC_ROUND(0 * 4 + 8, s, r);

		AES_ENC_ROUND(1 * 4, r, t);
		AES_ENC_ROUND(1 * 4 + 8, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 16, r, s);
		AES_ENC_ROUND(0 * 4 + 24, s, r);

		AES_ENC_ROUND(1 * 4 + 16, r, t);
		AES_ENC_ROUND(1 * 4 + 24, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 32, r, s);
		AES_ENC_ROUND(0 * 4 + 40, s, r);

		AES_ENC_ROUND(1 * 4 + 32, r, t);
		AES_ENC_ROUND(1 * 4 + 40, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 48, r, s);
		AES_ENC_ROUND(0 * 4 + 56, s, r);

		AES_ENC_ROUND(1 * 4 + 48, r, t);
		AES_ENC_ROUND(1 * 4 + 56, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 64, r, s);
		AES_ENC_ROUND(0 * 4 + 72, s, r);

		AES_ENC_ROUND(1 * 4 + 64, r, t);
		AES_ENC_ROUND(1 * 4 + 72, t, r);

		MIX_256

		load[0] ^= (s0 | ((uint64_t)s1) << 32);
		load[1] ^= (s2 | ((uint64_t)s3) << 32);
		load[2] ^= (t0 | ((uint64_t)t1) << 32);
		load[3] ^= (t2 | ((uint64_t)t3) << 32);

		if (i < 255)
		{
			s0 = load[0];
			s1 = load[0] >> 32;
			s2 = load[1];
			s3 = load[1] >> 32;

			t0 = load[2];
			t1 = load[2] >> 32;
			t2 = load[3];
			t3 = load[3] >> 32;
		}
	}

	if (b_ == 255)
	{
		reinterpret_cast<uint4*>(signature)[2 * TX] = reinterpret_cast<uint4*>(load)[0];
		reinterpret_cast<uint4*>(signature)[2 * TX + 1] = reinterpret_cast<uint4*>(load)[1];
	}

	reinterpret_cast<uint4*>(pub_key)[2 * TX] = reinterpret_cast<uint4*>(load)[0];
	reinterpret_cast<uint4*>(pub_key)[2 * TX + 1] = reinterpret_cast<uint4*>(load)[1];
}

__global__ void harakaOTSCreateVerificationKernel(const uint64_t* signature, uint64_t* verification, uint8_t* b, const uint32_t num_chunks)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_chunks))
		return;

	GLOBAL_LOAD_SHARED_SETUP_256(signature)

	uint8_t b_ = UINT8_MAX - b[TX];

	for (uint8_t i = 0; i < b_; ++i)
	{
		AES_ENC_ROUND(0 * 4, r, s);
		AES_ENC_ROUND(0 * 4 + 8, s, r);

		AES_ENC_ROUND(1 * 4, r, t);
		AES_ENC_ROUND(1 * 4 + 8, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 16, r, s);
		AES_ENC_ROUND(0 * 4 + 24, s, r);

		AES_ENC_ROUND(1 * 4 + 16, r, t);
		AES_ENC_ROUND(1 * 4 + 24, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 32, r, s);
		AES_ENC_ROUND(0 * 4 + 40, s, r);

		AES_ENC_ROUND(1 * 4 + 32, r, t);
		AES_ENC_ROUND(1 * 4 + 40, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 48, r, s);
		AES_ENC_ROUND(0 * 4 + 56, s, r);

		AES_ENC_ROUND(1 * 4 + 48, r, t);
		AES_ENC_ROUND(1 * 4 + 56, t, r);

		MIX_256

		AES_ENC_ROUND(0 * 4 + 64, r, s);
		AES_ENC_ROUND(0 * 4 + 72, s, r);

		AES_ENC_ROUND(1 * 4 + 64, r, t);
		AES_ENC_ROUND(1 * 4 + 72, t, r);

		MIX_256

		load[0] ^= (s0 | ((uint64_t)s1) << 32);
		load[1] ^= (s2 | ((uint64_t)s3) << 32);
		load[2] ^= (t0 | ((uint64_t)t1) << 32);
		load[3] ^= (t2 | ((uint64_t)t3) << 32);

		if (i < 255)
		{
			s0 = load[0];
			s1 = load[0] >> 32;
			s2 = load[1];
			s3 = load[1] >> 32;

			t0 = load[2];
			t1 = load[2] >> 32;
			t2 = load[3];
			t3 = load[3] >> 32;
		}
	}

	reinterpret_cast<uint4*>(verification)[2 * TX] = reinterpret_cast<uint4*>(load)[0];

	reinterpret_cast<uint4*>(verification)[2 * TX + 1] = reinterpret_cast<uint4*>(load)[1];
}


#define GLOBAL_LOAD_SHARED_SETUP_MERKLE \
		register uint32_t r0, r1, r2, r3; \
		register uint32_t s0, s1, s2, s3; \
		register uint32_t t0, t1, t2, t3; \
		register uint32_t u0, u1, u2, u3; \
		register uint32_t v0, v1, v2, v3; \
		register uint64_t load[2]; \
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(tree)[(childs - 1) * 2 + 4 * TX]; \
		s0 = load[0]; \
		s1 = load[0] >> 32; \
		s2 = load[1]; \
		s3 = load[1] >> 32; \
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(tree)[(childs - 1) * 2 + 4 * TX + 1]; \
		t0 = load[0]; \
		t1 = load[0] >> 32; \
		t2 = load[1]; \
		t3 = load[1] >> 32;\
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(tree)[(childs - 1) * 2 + 4 * TX + 2]; \
		u0 = load[0]; \
		u1 = load[0] >> 32; \
		u2 = load[1]; \
		u3 = load[1] >> 32;\
		reinterpret_cast<uint4*>(load)[0] = reinterpret_cast<const uint4*>(tree)[(childs - 1) * 2 + 4 * TX + 3]; \
		v0 = load[0]; \
		v1 = load[0] >> 32; \
		v2 = load[1]; \
		v3 = load[1] >> 32;

__global__ void harakaBuildMerkleTree(uint64_t* tree, const uint32_t num_parents, const uint32_t depth)
{
	COPY_CONSTANT_SHARED_ENC

	if (TX >= (num_parents))
		return;

	uint32_t childs = num_parents * 2;

	GLOBAL_LOAD_SHARED_SETUP_MERKLE

	for (int s = 1; s < (1 << depth); s *= 2)
	{
		if ((threadIdx.x % s) == 0)
		{
			AES_ENC_ROUND(0, r, s);
			AES_ENC_ROUND(16, s, r);

			AES_ENC_ROUND(1 * 4, r, t);
			AES_ENC_ROUND(1 * 4 + 16, t, r);

			AES_ENC_ROUND(2 * 4, r, u);
			AES_ENC_ROUND(2 * 4 + 16, u, r);

			AES_ENC_ROUND(3 * 4, r, v);
			AES_ENC_ROUND(3 * 4 + 16, v, r);

			MIX_512

			AES_ENC_ROUND(0 * 4 + 32, r, s);
			AES_ENC_ROUND(0 * 4 + 48, s, r);

			AES_ENC_ROUND(1 * 4 + 32, r, t);
			AES_ENC_ROUND(1 * 4 + 48, t, r);

			AES_ENC_ROUND(2 * 4 + 32, r, u);
			AES_ENC_ROUND(2 * 4 + 48, u, r);

			AES_ENC_ROUND(3 * 4 + 32, r, v);
			AES_ENC_ROUND(3 * 4 + 48, v, r);

			MIX_512

			AES_ENC_ROUND(0 * 4 + 64, r, s);
			AES_ENC_ROUND(0 * 4 + 80, s, r);

			AES_ENC_ROUND(1 * 4 + 64, r, t);
			AES_ENC_ROUND(1 * 4 + 80, t, r);

			AES_ENC_ROUND(2 * 4 + 64, r, u);
			AES_ENC_ROUND(2 * 4 + 80, u, r);

			AES_ENC_ROUND(3 * 4 + 64, r, v);
			AES_ENC_ROUND(3 * 4 + 80, v, r);

			MIX_512

			AES_ENC_ROUND(0 * 4 + 96, r, s);
			AES_ENC_ROUND(0 * 4 + 112, s, r);

			AES_ENC_ROUND(1 * 4 + 96, r, t);
			AES_ENC_ROUND(1 * 4 + 112, t, r);

			AES_ENC_ROUND(2 * 4 + 96, r, u);
			AES_ENC_ROUND(2 * 4 + 112, u, r);

			AES_ENC_ROUND(3 * 4 + 96, r, v);
			AES_ENC_ROUND(3 * 4 + 112, v, r);

			MIX_512

			AES_ENC_ROUND(0 * 4 + 128, r, s);
			AES_ENC_ROUND(0 * 4 + 144, s, r);

			AES_ENC_ROUND(1 * 4 + 128, r, t);
			AES_ENC_ROUND(1 * 4 + 144, t, r);

			AES_ENC_ROUND(2 * 4 + 128, r, u);
			AES_ENC_ROUND(2 * 4 + 144, u, r);

			AES_ENC_ROUND(3 * 4 + 128, r, v);
			AES_ENC_ROUND(3 * 4 + 144, v, r);

			MIX_512

			load[1] = s2 | ((uint64_t)s3) << 32;
			load[1] ^= tree[(childs - 1) * 4 + 8 * TX / s + 1];
			tree[(childs / 2 - 1) * 4 + TX / s * 4] = load[1];
			s0 = load[1];
			s1 = load[1] >> 32;

			load[1] = t2 | ((uint64_t)t3) << 32;
			load[1] ^= tree[(childs - 1) * 4 + 8 * TX / s + 3];
			tree[(childs / 2 - 1) * 4 + TX / s * 4 + 1] = load[1];
			s2 = load[1];
			s3 = load[1] >> 32;

			load[1] = u0 | ((uint64_t)u1) << 32;
			load[1] ^= tree[(childs - 1) * 4 + 8 * TX / s + 4];
			tree[(childs / 2 - 1) * 4 + TX / s * 4 + 2] = load[1];
			t0 = load[1];
			t1 = load[1] >> 32;

			load[1] = v0 | ((uint64_t)v1) << 32;
			load[1] ^= tree[(childs - 1) * 4 + 8 * TX / s + 6];
			tree[(childs / 2 - 1) * 4 + TX / s * 4 + 3] = load[1];
			t2 = load[1];
			t3 = load[1] >> 32;

			u0 = __shfl_down(s0, s);
			u1 = __shfl_down(s1, s);
			u2 = __shfl_down(s2, s);
			u3 = __shfl_down(s3, s);
			v0 = __shfl_down(t0, s);
			v1 = __shfl_down(t1, s);
			v2 = __shfl_down(t2, s);
			v3 = __shfl_down(t3, s);

			childs = childs >> 1;
		}
	}
}